"""
infer_stage2.py
===============
Stage 2 Inference: Image Features → Synthetic fMRI → Reconstructed Images

Pipeline:
  1. Load DINOv2 features for given test indices
  2. Generate synthetic fMRI via BrainMaskedDiT + VAE
  3. Decode fMRI → 2D images via MindEye2 pipeline
  4. Plot: Original Image | Raw (Stage 2) | Enhanced (Stage 3)

Usage:
  python src/infer_stage2.py --json_input '[ {"test_index": 115, "captions": ["An airplane..."]} ]'
  python src/infer_stage2.py --json_file Data/evals/eval_input.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.brain_masked_dit import BrainMaskedDiT, BrainMaskedDiTConfig
from src.model.fmri_vit_vae import FmriViTVAE, create_fmri_vit_vae
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig


def load_vae(vae_ckpt_path, device):
    """Load VAE model from checkpoint, auto-detecting config."""
    vae_dir = os.path.dirname(vae_ckpt_path)
    config_path = os.path.join(vae_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"VAE config not found at {config_path}")

    with open(config_path) as f:
        vae_cfg = yaml.safe_load(f)

    if vae_cfg.get("model_type", "mlp") == "vit":
        vae = create_fmri_vit_vae(**vae_cfg["model"])
    else:
        vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"]))

    ckpt = torch.load(vae_ckpt_path, map_location='cpu')
    vae.load_state_dict(ckpt["model_state_dict"])
    vae = vae.to(device).eval()
    print(f"VAE loaded: {vae_ckpt_path}")
    return vae


def load_stage2_model(stage2_ckpt_path, model_cfg, device):
    """Load BrainMaskedDiT model."""
    config = BrainMaskedDiTConfig(**model_cfg)
    model = BrainMaskedDiT(config).to(device)

    ckpt = torch.load(stage2_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"BrainMaskedDiT loaded: {stage2_ckpt_path}")
    return model


def generate_fmri(model, vae, dino_features, device):
    """
    Generate synthetic fMRI from DINOv2 features.
    
    Args:
        model: BrainMaskedDiT model
        vae: FmriViTVAE model
        dino_features: np.ndarray (4, 257, 768) for a single sample
        device: torch device
    
    Returns:
        fmri_pred: np.ndarray (15724,)
    """
    with torch.no_grad():
        dino_tensor = torch.from_numpy(dino_features).float().unsqueeze(0).to(device)  # (1, 4, 257, 768)
        
        # Create dummy z_t (will be fully masked, so content doesn't matter)
        latent_dim = model.config.latent_dim
        dummy_z = torch.zeros(1, latent_dim, device=device)
        
        # Generate latent with full masking (mask_ratio=1.0)
        z_pred, _ = model(dummy_z, dino_tensor, mask_ratio=1.0)
        
        # Decode latent → fMRI
        fmri_pred = vae.decode(z_pred)
        
    return fmri_pred.cpu().numpy().squeeze()  # (15724,)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Inference: Image → fMRI → Reconstructed Image")
    parser.add_argument("--json_input", type=str, default=None,
                        help="JSON string with list of {test_index, captions}")
    parser.add_argument("--json_file", type=str, default=None,
                        help="Path to JSON file with list of {test_index, captions}")
    parser.add_argument("--stage2_config", type=str,
                        default="results/subj01/stage2_masked_vit_vae/config.yaml")
    parser.add_argument("--stage2_ckpt", type=str,
                        default="results/subj01/stage2_masked_vit_vae/best.pt")
    parser.add_argument("--output_dir", type=str, default="Data/evals/stage2_infer")
    parser.add_argument("--cache_dir", type=str, default="Data/checkpoints",
                        help="MindEye2 checkpoint directory")
    parser.add_argument("--img2img_strength", type=int, default=13)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_caption", action="store_true",
                        help="Force all captions to be empty (unconditional generation)")
    parser.add_argument("--auto_caption", action="store_true",
                        help="Use MindEye2 Stage 1.5 to auto-generate captions from synthetic fMRI")
    parser.add_argument("--nii_dir", type=str, default=None,
                        help="Directory with synthetic test fMRI NIfTI files (e.g., Data/evals/SonFMRI). If set, image-to-fMRI generation is skipped.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Parse JSON input ----
    if args.json_input:
        samples = json.loads(args.json_input)
    elif args.json_file:
        with open(args.json_file) as f:
            samples = json.load(f)
    else:
        parser.error("Must provide either --json_input or --json_file")

    print(f"Processing {len(samples)} samples")

    # ---- Load Stage 2 config ----
    with open(args.stage2_config) as f:
        stage2_cfg = yaml.safe_load(f)

    data_cfg = stage2_cfg["data"]
    model_cfg = stage2_cfg.get("model", {})

    # ---- Load models (only if NOT using nii_dir) ----
    if not args.nii_dir:
        vae = load_vae(data_cfg["vae_checkpoint"], device)
        model = load_stage2_model(args.stage2_ckpt, model_cfg, device)

        # ---- Load DINOv2 test features (memory-mapped) ----
        sub_num = int(data_cfg["subject"].replace("subj", "").lstrip("0"))
        dino_suffix = data_cfg.get("dino_suffix", "dinov2_vitb14_multilayer")
        dino_path = os.path.join(data_cfg["root"], data_cfg["subject"],
                                 f"nsd_{dino_suffix}_test_sub{sub_num}.npy")
        dino_mmap = np.load(dino_path, mmap_mode='r')
        print(f"DINOv2 features: {dino_mmap.shape}")

    # ---- Load captions and test images ----
    sub_num = int(data_cfg["subject"].replace("subj", "").lstrip("0"))
    caps_path = os.path.join(data_cfg["root"], data_cfg["subject"],
                             f"nsd_test_cap_sub{sub_num}.npy")
    caps_test = np.load(caps_path)
    test_img_dir = os.path.join(data_cfg["root"], data_cfg["subject"], "test_img")

    # ---- Generate or Load fMRI for each sample ----
    all_fmri = []
    all_captions = []
    all_test_indices = []

    for sample in samples:
        test_idx = sample["test_index"]

        # Fetch fMRI
        if args.nii_dir:
            import nibabel as nib
            # Use eval_fmri field if available (e.g. "115_pred.nii")
            # Otherwise fallback to test_index based naming
            if "eval_fmri" in sample:
                nii_path = os.path.join(args.nii_dir, sample["eval_fmri"])
            else:
                nii_path = os.path.join(args.nii_dir, f"{test_idx}_pred.nii")
                if not os.path.exists(nii_path):
                    nii_path = os.path.join(args.nii_dir, f"{test_idx}_gt.nii")
            
            if not os.path.exists(nii_path):
                print(f"  [{test_idx}] Skipping: missing {nii_path}")
                continue
            
            # Load and pad to 15724 length just like infer_mindeye2 does
            nii_data = nib.load(nii_path).get_fdata().flatten().astype(np.float32)
            if len(nii_data) < 15724:
                fmri_pred = np.pad(nii_data, (0, 15724 - len(nii_data)), 'constant')
            else:
                fmri_pred = nii_data[:15724]
            
            print(f"  [{test_idx}] Loaded fMRI from {nii_path}: shape={fmri_pred.shape}")
        else:
            dino_feat = np.array(dino_mmap[test_idx])
            fmri_pred = generate_fmri(model, vae, dino_feat, device)
            print(f"  [{test_idx}] fMRI generated: shape={fmri_pred.shape}, "
                  f"range=[{fmri_pred.min():.3f}, {fmri_pred.max():.3f}]")
                  
        all_fmri.append(fmri_pred)
        all_test_indices.append(test_idx)

        # Get caption: from JSON input or fallback to saved captions
        if args.auto_caption:
            caption = None  # None tells mindeye2_reconstruct to run Stage 1.5
        elif args.no_caption:
            caption = ""
        elif "captions" in sample and sample["captions"]:
            caption = sample["captions"][0]
        else:
            caps = [c for c in caps_test[test_idx] if c.strip()]
            caption = caps[0] if caps else ""
        
        all_captions.append(caption)

    # Stack fMRI predictions
    fmri_stack = np.stack(all_fmri, axis=0)  # (N, 15724)
    print(f"\nSynthetic fMRI batch: {fmri_stack.shape}")

    # Save synthetic fMRI
    fmri_save_path = os.path.join(args.output_dir, "synthetic_fmri.npy")
    np.save(fmri_save_path, fmri_stack)
    print(f"Saved: {fmri_save_path}")

    # ---- Decode fMRI → Images via MindEye2 ----
    if True:
        print("\n--- Running MindEye2 Decoding ---")
        # Setup MindEye2 paths before importing
        mindeye_src = os.path.join(PROJECT_ROOT, 'Data', 'notes', 'MindEyeV2', 'src')
        mindeye_sgm = os.path.join(mindeye_src, 'generative_models')
        for p in [mindeye_src, mindeye_sgm]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from src.utils.infer_mindeye2 import reconstruct as mindeye2_reconstruct
        recon_results = mindeye2_reconstruct(
            fmri_data=fmri_stack,
            captions=all_captions,
            output_dir=os.path.join(args.output_dir, "mindeye2_output"),
            cache_dir=args.cache_dir,
            img2img_strength=args.img2img_strength,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )

        raw_images = recon_results['raw_images']
        enhanced_images = recon_results['enhanced_images']

        # ---- Plot: Original | Raw | Enhanced ----
        n = len(samples)
        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
        if n == 1:
            axes = [axes]

        for i, sample in enumerate(samples):
            test_idx = sample["test_index"]
            caption = recon_results['captions'][i] if recon_results['captions'] and len(recon_results['captions']) > i else all_captions[i]
            if caption is None: caption = ""

            # Col 1: Original test image
            orig_path = os.path.join(test_img_dir, f"{test_idx}.png")
            if os.path.exists(orig_path):
                orig_img = Image.open(orig_path)
            else:
                orig_img = Image.new('RGB', (425, 425), (200, 200, 200))

            display_id = sample.get("eval_image", f"Test #{test_idx}")
            if display_id.endswith(".png"): display_id = display_id[:-4]

            axes[i][0].imshow(orig_img)
            axes[i][0].set_title(
                f"Original ({display_id})\n{caption[:100]}",
                fontsize=9, loc='left'
            )
            axes[i][0].axis('off')

            # Col 2: Raw image (MindEye2 Stage 2 unCLIP)
            axes[i][1].imshow(raw_images[i])
            axes[i][1].set_title("MindEye2 Raw (unCLIP)", fontsize=10)
            axes[i][1].axis('off')

            # Col 3: Enhanced image (MindEye2 Stage 3 SDXL)
            axes[i][2].imshow(enhanced_images[i])
            axes[i][2].set_title("MindEye2 Enhanced (SDXL)", fontsize=10)
            axes[i][2].axis('off')

        plt.suptitle("Stage 2 Inference: Original → Raw → Enhanced", fontsize=14, y=1.01)
        plt.tight_layout()

        plot_path = os.path.join(args.output_dir, "comparison_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved: {plot_path}")

    # ---- Save JSON results ----
    output_json = []
    for i, sample in enumerate(samples):
        cpt = recon_results['captions'][i] if 'recon_results' in locals() and recon_results['captions'] and len(recon_results['captions']) > i else all_captions[i]
        entry = {
            "test_index": sample["test_index"],
            "caption": cpt,
            "fmri_stats": {
                "mean": float(all_fmri[i].mean()),
                "std": float(all_fmri[i].std()),
                "min": float(all_fmri[i].min()),
                "max": float(all_fmri[i].max()),
            }
        }
        output_json.append(entry)

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"\nResults JSON: {json_path}")
    print(json.dumps(output_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
