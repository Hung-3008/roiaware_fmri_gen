"""
infer_mindeye2.py
=================
MindEye2 3-Stage fMRI → Image Reconstruction Pipeline

Accepts fMRI input as:
  - numpy array (N, 15724) or (N, 3, 15724) passed programmatically
  - .npy file via --fmri_npy
  - directory of .nii files via --nii_dir (legacy GT/Pred pair mode)

Outputs reconstructed images (PNG) and captions (JSON) to --output_dir.

Stages:
  1.  fMRI → CLIP latents  (Ridge → BrainNetwork → DiffusionPrior)
  1.5 CLIP latents → Captions  (bigG→L converter + GitForCausalLM)
  2.  CLIP latents → Raw image  (unCLIP DiffusionEngine)
  3.  Raw image + Caption → Enhanced image  (SDXL img2img refinement)

Usage:
  # From .npy file
  python src/utils/infer_mindeye2.py --fmri_npy Data/nsd/subj01/nsd_test_fmri_zscore_sub1.npy --output_dir evals/recon/

  # From legacy NIfTI pairs
  python src/utils/infer_mindeye2.py --nii_dir evals/infer_ress/ --output_dir evals/recon/
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import gc

# ==============================================================================
# Path setup for MindEye2 dependencies
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MINDEYE_SRC = os.path.join(PROJECT_ROOT, 'Data', 'notes', 'MindEyeV2', 'src')
MINDEYE_SGM = os.path.join(MINDEYE_SRC, 'generative_models')

for p in [MINDEYE_SRC, MINDEYE_SGM]:
    if p not in sys.path:
        sys.path.insert(0, p)

from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf
import utils as me_utils
from models import BrainNetwork, BrainDiffusionPrior, PriorNetwork

torch.backends.cuda.matmul.allow_tf32 = True

# ==============================================================================
# Constants
# ==============================================================================
NUM_VOXELS = 15724
CLIP_SEQ_DIM = 256
CLIP_EMB_DIM = 1664

# ==============================================================================
# Helper models (same as original)
# ==============================================================================
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList([
            nn.Linear(input_size, out_features) for input_size in input_sizes
        ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out

class CLIPConverter(nn.Module):
    """Converts CLIP bigG embeddings to CLIP-L space for captioning."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(CLIP_SEQ_DIM, 257)
        self.linear2 = nn.Linear(CLIP_EMB_DIM, 1024)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x


# ==============================================================================
# Input adapter: normalize any fMRI input to List[np.ndarray] of shape (V,)
# ==============================================================================
def load_fmri_inputs(fmri_npy=None, nii_dir=None, fmri_data=None):
    """
    Accepts fMRI data from multiple sources and returns a list of 1D arrays (V,).

    Args:
        fmri_npy: path to .npy file, shape (N, V) or (N, 3, V)
        nii_dir: path to directory with *_gt.nii / *_pred.nii pairs (legacy)
        fmri_data: numpy array or torch tensor, shape (N, V) or (N, 3, V) or (V,)

    Returns:
        samples: list of np.ndarray each with shape (NUM_VOXELS,)
        labels: list of str (sample identifiers)
    """
    samples = []
    labels = []

    # Priority 1: Direct tensor/array input
    if fmri_data is not None:
        if isinstance(fmri_data, torch.Tensor):
            fmri_data = fmri_data.detach().cpu().numpy()
        if fmri_data.ndim == 1:
            fmri_data = fmri_data[np.newaxis, :]  # (V,) -> (1, V)
        if fmri_data.ndim == 3:
            # (N, 3, V) -> take mean of 3 repetitions -> (N, V)
            fmri_data = fmri_data.mean(axis=1)
        for i in range(fmri_data.shape[0]):
            vec = fmri_data[i].astype(np.float32)
            vec = _pad_or_truncate(vec, NUM_VOXELS)
            samples.append(vec)
            labels.append(f"sample_{i:04d}")
        return samples, labels

    # Priority 2: .npy file
    if fmri_npy is not None:
        arr = np.load(fmri_npy)
        return load_fmri_inputs(fmri_data=arr)

    # Priority 3: NIfTI directory (legacy GT/Pred pairs)
    if nii_dir is not None:
        import nibabel as nib
        gt_files = sorted(glob.glob(os.path.join(nii_dir, "*_gt.nii*")))
        for gt_file in gt_files:
            basename = os.path.basename(gt_file)
            sample_id = basename.split("_gt.nii")[0]
            pred_file = os.path.join(nii_dir, basename.replace('_gt.', '_pred.'))
            mean_pred_file = os.path.join(nii_dir, basename.replace('_gt.', '_mean_pred.'))

            # Load GT
            gt_data = nib.load(gt_file).get_fdata().flatten().astype(np.float32)
            gt_data = _pad_or_truncate(gt_data, NUM_VOXELS)
            samples.append(gt_data)
            labels.append(f"{sample_id}_gt")

            # Load Pred if exists
            if os.path.exists(pred_file):
                pred_data = nib.load(pred_file).get_fdata().flatten().astype(np.float32)
                pred_data = _pad_or_truncate(pred_data, NUM_VOXELS)
                samples.append(pred_data)
                labels.append(f"{sample_id}_pred")
            elif os.path.exists(mean_pred_file):
                pred_data = nib.load(mean_pred_file).get_fdata().flatten().astype(np.float32)
                pred_data = _pad_or_truncate(pred_data, NUM_VOXELS)
                samples.append(pred_data)
                labels.append(f"{sample_id}_mean_pred")

        return samples, labels

    raise ValueError("Must provide one of: fmri_data, fmri_npy, or nii_dir")


def _pad_or_truncate(vec, target_len):
    """Ensure vector has exactly target_len elements."""
    if len(vec) > target_len:
        return vec[:target_len]
    elif len(vec) < target_len:
        return np.pad(vec, (0, target_len - len(vec)))
    return vec


# ==============================================================================
# STAGE 1: fMRI → CLIP latents
# ==============================================================================
def run_stage1(fmri_samples, cache_dir, device, hidden_dim=4096):
    """
    Args:
        fmri_samples: list of np.ndarray, each shape (NUM_VOXELS,)
        cache_dir: path to checkpoint directory
        device: torch device

    Returns:
        prior_outs: list of torch.Tensor, each shape (1, 256, 1664) on CPU
    """
    print("\n--- STAGE 1: MindEye2 fMRI → CLIP Latents ---")

    model = MindEyeModule()
    model.ridge = RidgeRegression([NUM_VOXELS], out_features=hidden_dim)
    model.backbone = BrainNetwork(
        h=hidden_dim, in_dim=hidden_dim, seq_len=1,
        clip_size=CLIP_EMB_DIM, out_dim=CLIP_EMB_DIM * CLIP_SEQ_DIM
    )

    prior_network = PriorNetwork(
        dim=CLIP_EMB_DIM, depth=6, dim_head=52, heads=CLIP_EMB_DIM // 52,
        causal=False, num_tokens=CLIP_SEQ_DIM, learned_query_mode="pos_emb"
    )
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network, image_embed_dim=CLIP_EMB_DIM,
        condition_on_text_encodings=False,
        timesteps=100, cond_drop_prob=0.2, image_embed_scale=None,
    )

    ckpt_path = os.path.join(cache_dir, 'mindeye2', 'last.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MindEye2 checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval().requires_grad_(False).to(device)
    print(f"  Loaded: {ckpt_path}")

    prior_outs = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i, fmri_vec in enumerate(tqdm(fmri_samples, desc="Stage 1")):
            fmri_tensor = torch.tensor(fmri_vec, dtype=torch.float32).to(device)
            fmri_tensor = fmri_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, V)

            voxel_ridge = model.ridge(fmri_tensor, 0)
            backbone_out, _, _ = model.backbone(voxel_ridge)
            prior_out = model.diffusion_prior.p_sample_loop(
                backbone_out.shape, text_cond=dict(text_embed=backbone_out),
                cond_scale=1., timesteps=20
            )
            prior_outs.append(prior_out.cpu())

    del model, checkpoint
    torch.cuda.empty_cache()
    gc.collect()
    return prior_outs


# ==============================================================================
# STAGE 1.5: CLIP latents → Captions
# ==============================================================================
def run_stage1_5(prior_outs, cache_dir, device):
    """
    Args:
        prior_outs: list of torch.Tensor, each (1, 256, 1664)
        cache_dir: path to checkpoint directory
        device: torch device

    Returns:
        captions: list of str
    """
    print("\n--- STAGE 1.5: Generate Captions ---")

    from transformers import AutoProcessor
    from transformers.generation import GenerationMixin
    from modeling_git import GitForCausalLMClipEmb

    if GenerationMixin not in GitForCausalLMClipEmb.__mro__:
        GitForCausalLMClipEmb.__bases__ = (GenerationMixin,) + GitForCausalLMClipEmb.__bases__

    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    clip_text_model.to(device).eval().requires_grad_(False)

    clip_convert = CLIPConverter()
    cc_ckpt_path = os.path.join(cache_dir, 'bigG_to_L_epoch8.pth')
    if os.path.exists(cc_ckpt_path):
        cc_state = torch.load(cc_ckpt_path, map_location='cpu')['model_state_dict']
        clip_convert.load_state_dict(cc_state, strict=True)
        del cc_state
        print(f"  Loaded: {cc_ckpt_path}")
    else:
        print(f"  Warning: {cc_ckpt_path} not found, using random weights")

    clip_convert.to(device)

    captions = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i in tqdm(range(len(prior_outs)), desc="Stage 1.5"):
            caption_emb = clip_convert(prior_outs[i].to(device))
            generated_ids = clip_text_model.generate(
                pixel_values=caption_emb, max_length=20
            )
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"DEBUG caption type: {type(caption)}, value: '{caption}'")
            captions.append(str(caption).strip())

    del clip_text_model, clip_convert, processor
    torch.cuda.empty_cache()
    gc.collect()
    return captions


# ==============================================================================
# STAGE 2: CLIP latents → Raw images via unCLIP
# ==============================================================================
def run_stage2(prior_outs, cache_dir, device):
    """
    Args:
        prior_outs: list of torch.Tensor, each (1, 256, 1664)
        cache_dir: path to checkpoint directory
        device: torch device

    Returns:
        raw_images: list of np.ndarray, each (224, 224, 3) in [0, 1]
    """
    print("\n--- STAGE 2: unCLIP Reconstruction ---")

    config = OmegaConf.load(os.path.join(MINDEYE_SGM, "configs", "unclip6.yaml"))
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    unclip_params['first_stage_config']['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    unclip_params['sampler_config']['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(
        network_config=unclip_params["network_config"],
        denoiser_config=unclip_params["denoiser_config"],
        first_stage_config=unclip_params["first_stage_config"],
        conditioner_config=unclip_params["conditioner_config"],
        sampler_config=unclip_params["sampler_config"],
        scale_factor=unclip_params["scale_factor"],
        disable_first_stage_autocast=unclip_params["disable_first_stage_autocast"]
    )

    unclip_ckpt_path = os.path.join(cache_dir, 'sdxl_unclip', 'unclip6_epoch0_step110000.ckpt')
    if not os.path.exists(unclip_ckpt_path):
        raise FileNotFoundError(f"unCLIP checkpoint not found: {unclip_ckpt_path}")

    ckpt = torch.load(unclip_ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    diffusion_engine.eval().requires_grad_(False).to(device)
    print(f"  Loaded: {unclip_ckpt_path}")

    # Compute vector suffix (same as original)
    batch_ph = {
        "jpg": torch.randn(1, 3, 1, 1).to(device),
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device)
    }
    vector_suffix = diffusion_engine.conditioner(batch_ph)["vector"].to(device)

    raw_images = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i in tqdm(range(len(prior_outs)), desc="Stage 2"):
            prior_out = prior_outs[i].to(device)
            recon_ts = me_utils.unclip_recon(
                prior_out, diffusion_engine, vector_suffix, num_samples=1
            )
            recon_img = transforms.Resize((224, 224))(recon_ts).float().cpu()
            recon_img = recon_img[0].permute(1, 2, 0).numpy()
            recon_img = np.clip(recon_img, 0, 1)
            raw_images.append(recon_img)

    # Store unclip_params for Stage 3 (sampler_config reuse)
    run_stage2._unclip_params = unclip_params

    del diffusion_engine, ckpt
    torch.cuda.empty_cache()
    gc.collect()
    return raw_images


# ==============================================================================
# STAGE 3: Raw image + Caption → Enhanced image via SDXL img2img
# ==============================================================================
def run_stage3(raw_images, captions, cache_dir, device, img2img_strength=13, cfg_scale=5.0):
    """
    Args:
        raw_images: list of np.ndarray, each (224, 224, 3) in [0, 1]
        captions: list of str
        cache_dir: path to checkpoint directory
        device: torch device
        img2img_strength: denoising unclip step (default 13)
        cfg_scale: classifier-free guidance scale (default 5.0)

    Returns:
        enhanced_images: list of np.ndarray, each (224, 224, 3) in [0, 1]
    """
    print("\n--- STAGE 3: SDXL Refinement ---")

    # Monkey-patch OpenCLIP to avoid PyTorch 2.1+ attn_mask shape error
    import open_clip
    old_attention = open_clip.transformer.ResidualAttentionBlock.attention
    def new_attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
        try:
            return old_attention(self, q_x, k_x, v_x, attn_mask)
        except RuntimeError:
            return self.attn(
                q_x, k_x if k_x is not None else q_x,
                v_x if v_x is not None else q_x,
                need_weights=False, attn_mask=None
            )[0]
    open_clip.transformer.ResidualAttentionBlock.attention = new_attention

    sdxl_config = OmegaConf.load(os.path.join(MINDEYE_SGM, "configs", "inference", "sd_xl_base.yaml"))
    sdxl_config = OmegaConf.to_container(sdxl_config, resolve=True)
    refiner_params = sdxl_config["model"]["params"]

    # Patch conditioner to use local OpenCLIP checkpoint instead of downloading
    open_clip_local = os.path.abspath(os.path.join(cache_dir, 'sdxl_unclip', 'open_clip_pytorch_model.bin'))
    if os.path.exists(open_clip_local):
        for emb_cfg in refiner_params["conditioner_config"]["params"]["emb_models"]:
            if emb_cfg.get("target", "").endswith("FrozenOpenCLIPEmbedder2"):
                emb_cfg["params"]["version"] = open_clip_local
                print(f"  Patched FrozenOpenCLIPEmbedder2 to use local: {open_clip_local}")

    # Reuse unclip sampler_config from Stage 2
    unclip_params = getattr(run_stage2, '_unclip_params', None)
    if unclip_params is None:
        config = OmegaConf.load(os.path.join(MINDEYE_SGM, "configs", "unclip6.yaml"))
        config = OmegaConf.to_container(config, resolve=True)
        unclip_params = config["model"]["params"]

    base_ckpt_path = os.path.join(cache_dir, 'zavychromaxl_v30.safetensors')
    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(f"SDXL checkpoint not found: {base_ckpt_path}")

    base_engine = DiffusionEngine(
        network_config=refiner_params["network_config"],
        denoiser_config=refiner_params["denoiser_config"],
        first_stage_config=refiner_params["first_stage_config"],
        conditioner_config=refiner_params["conditioner_config"],
        sampler_config=unclip_params["sampler_config"],
        scale_factor=refiner_params["scale_factor"],
        disable_first_stage_autocast=refiner_params["disable_first_stage_autocast"],
        ckpt_path=base_ckpt_path
    )
    base_engine.eval().requires_grad_(False).to(device)
    print(f"  Loaded: {base_ckpt_path}")

    # Unconditional embeddings
    batch_uc = {
        "txt": "",
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
        "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024
    }
    out_uc = base_engine.conditioner(batch_uc)
    crossattn_uc = out_uc["crossattn"].to(device)
    vector_uc = out_uc["vector"].to(device)

    def denoiser(x, sigma, c):
        return base_engine.denoiser(base_engine.model, x, sigma, c)

    img2img_timepoint = img2img_strength

    def enhance_img(raw_img_np, prompt):
        base_engine.sampler.num_steps = 25
        base_engine.sampler.guider.scale = cfg_scale

        raw_img = torch.tensor(raw_img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        raw_img = transforms.Resize((768, 768))(raw_img).float()

        batch_c = {
            "txt": prompt,
            "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2).to(device),
            "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024
        }
        out_c = base_engine.conditioner(batch_c)
        c = {"crossattn": out_c["crossattn"].to(device), "vector": out_c["vector"].to(device)}
        uc = {"crossattn": crossattn_uc, "vector": vector_uc}

        z = base_engine.encode_first_stage(raw_img * 2 - 1)
        noise = torch.randn_like(z)
        sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
        init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )
        sigmas = sigmas[-img2img_timepoint:].repeat(1, 1)

        base_engine.sampler.num_steps = sigmas.shape[-1] - 1
        noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
            init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps
        )
        for timestep in range(base_engine.sampler.num_steps):
            noised_z = base_engine.sampler.sampler_step(
                sigmas[:, timestep], sigmas[:, timestep + 1],
                denoiser, noised_z, cond=c, uc=uc, gamma=0
            )

        samples_x = base_engine.decode_first_stage(noised_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        return transforms.Resize((224, 224))(samples[0]).cpu().permute(1, 2, 0).numpy()

    enhanced_images = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16), base_engine.ema_scope():
        for i in tqdm(range(len(raw_images)), desc="Stage 3"):
            enh = enhance_img(raw_images[i], captions[i])
            enhanced_images.append(enh)

    del base_engine
    torch.cuda.empty_cache()
    gc.collect()
    return enhanced_images


# ==============================================================================
# Full pipeline: fMRI → Enhanced Images
# ==============================================================================
def reconstruct(fmri_data=None, fmri_npy=None, nii_dir=None,
                captions=None,
                output_dir="evals/recon", cache_dir="Data/checkpoints",
                hidden_dim=4096, seed=42, device=None,
                img2img_strength=13, cfg_scale=5.0):
    """
    End-to-end MindEye2 reconstruction pipeline.

    Args:
        fmri_data: np.ndarray or torch.Tensor, shape (N, V) or (N, 3, V) or (V,)
        fmri_npy: path to .npy file
        nii_dir: path to directory with NIfTI files
        captions: optional list of str. If provided, Stage 1.5 is skipped.
                  Length must match the number of fMRI samples.
        output_dir: directory to save output images and captions
        cache_dir: directory containing model checkpoints
        hidden_dim: MindEye2 hidden dimension (default 4096)
        seed: random seed
        device: torch device (default: auto-detect)
        img2img_strength: denoising unclip step (default 13)
        cfg_scale: classifier-free guidance scale (default 5.0)

    Returns:
        dict with keys:
            'raw_images': list of np.ndarray (224, 224, 3)
            'enhanced_images': list of np.ndarray (224, 224, 3)
            'captions': list of str
            'labels': list of str (sample identifiers)
    """
    me_utils.seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load inputs
    samples, labels = load_fmri_inputs(
        fmri_npy=fmri_npy, nii_dir=nii_dir, fmri_data=fmri_data
    )
    print(f"Loaded {len(samples)} fMRI samples")

    # Stage 1: fMRI → CLIP latents
    prior_outs = run_stage1(samples, cache_dir, device, hidden_dim)

    # Stage 1.5: CLIP latents → Captions (skip if captions already provided)
    if captions is not None and not all(c is None for c in captions):
        assert len(captions) == len(samples), \
            f"Number of captions ({len(captions)}) must match number of fMRI samples ({len(samples)})"
        print(f"\n--- STAGE 1.5: Skipped (using {len(captions)} provided captions) ---")
    else:
        captions = run_stage1_5(prior_outs, cache_dir, device)

    # Stage 2: CLIP latents → Raw images
    raw_images = run_stage2(prior_outs, cache_dir, device)

    # Stage 3: Raw images → Enhanced images
    enhanced_images = run_stage3(raw_images, captions, cache_dir, device, img2img_strength, cfg_scale)

    # Save outputs
    for i, label in enumerate(labels):
        # Raw image
        raw_path = os.path.join(output_dir, f"{label}_raw.png")
        Image.fromarray((raw_images[i] * 255).astype(np.uint8)).save(raw_path)

        # Enhanced image
        enh_path = os.path.join(output_dir, f"{label}_enhanced.png")
        Image.fromarray((enhanced_images[i] * 255).astype(np.uint8)).save(enh_path)

    # Save captions
    caption_dict = {label: cap for label, cap in zip(labels, captions)}
    caption_path = os.path.join(output_dir, "captions.json")
    with open(caption_path, "w") as f:
        json.dump(caption_dict, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(labels)} reconstructions to {output_dir}")
    print(f"  Raw images: *_raw.png")
    print(f"  Enhanced images: *_enhanced.png")
    print(f"  Captions: captions.json")

    return {
        'raw_images': raw_images,
        'enhanced_images': enhanced_images,
        'captions': captions,
        'labels': labels,
    }


# ==============================================================================
# CLI entry point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MindEye2 fMRI → Image Reconstruction")
    parser.add_argument("--fmri_npy", type=str, default=None,
                        help="Path to .npy file with fMRI data, shape (N, V) or (N, 3, V)")
    parser.add_argument("--nii_dir", type=str, default=None,
                        help="Path to directory with *_gt.nii / *_pred.nii pairs")
    parser.add_argument("--captions_npy", type=str, default=None,
                        help="Path to .npy file with captions, shape (N,) or (N, K). "
                             "If provided, Stage 1.5 caption generation is skipped.")
    parser.add_argument("--output_dir", type=str, default="evals/recon")
    parser.add_argument("--cache_dir", type=str, default="Data/checkpoints")
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img2img_strength", type=int, default=13,
                        help="Denoising steps for MindEye Stage 3 (default: 13, lower = closer to CLIP raw image)")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale for caption conditioning (default: 5.0, lower = less bias to caption)")
    args = parser.parse_args()

    if args.fmri_npy is None and args.nii_dir is None:
        parser.error("Must specify either --fmri_npy or --nii_dir")

    # Load captions from file if provided
    captions = None
    if args.captions_npy is not None:
        cap_arr = np.load(args.captions_npy)
        if cap_arr.ndim == 2:
            captions = [cap_arr[i, 0] for i in range(cap_arr.shape[0])]
        else:
            captions = list(cap_arr)
        print(f"Loaded {len(captions)} captions from {args.captions_npy}")

    reconstruct(
        fmri_npy=args.fmri_npy,
        nii_dir=args.nii_dir,
        captions=captions,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        img2img_strength=args.img2img_strength,
        cfg_scale=args.cfg_scale,
    )


if __name__ == "__main__":
    main()
