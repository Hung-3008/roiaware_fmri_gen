"""
BrainFlow V2 — FlowFM-Style Training Script.

Direct flow matching from N(0,I) to fMRI voxel space, conditioned on
raw images via a trainable ViT encoder (FlowFM architecture).

Key differences from train_brainflow.py (V1):
  - Loads raw images instead of pre-extracted DINOv2 features
  - Jointly trains ViT encoder + velocity network
  - Separate learning rates for encoder vs velocity
  - DGS operates on representation r (zero-out ViT output)

Usage:
    python -m src.train_brainflow_v2 --config src/configs/subj01/brainflow_v2.yaml
    python -m src.train_brainflow_v2 --config src/configs/subj01/brainflow_v2.yaml --debug
"""

import argparse
import copy
import csv
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from src.model.brain_flow_v2 import (
    BrainFlowV2, BrainFlowV2Config, timestep_embedding, modulate,
)
from src.utils.roi_utils import ROIDecomposer, build_roi_perm


# ─── Dataset ──────────────────────────────────────────────────────────────────


class FmriImageDataset(Dataset):
    """Dataset pairing fMRI with raw stimulus images.

    Supports two modes:
      - Averaged mode (default): fMRI (N, R, V) → mean over R → (N, V)
      - Single-trial mode: fMRI (N, R, V) → expand to (N*R, V), each trial
        maps back to its original image index. ~3x more training data.
    """

    def __init__(self, fmri_path, stim_path, image_dir,
                 split="train", image_size=224, max_samples=0,
                 use_single_trials=False):
        print(f"\nFmriImageDataset [{split}]: Loading...")

        # Load fMRI
        raw_fmri = np.load(fmri_path)
        if raw_fmri.ndim == 3 and use_single_trials:
            # Single-trial mode: (N, R, V) → (N*R, V)
            N, R, V = raw_fmri.shape
            fmri = raw_fmri.reshape(N * R, V).astype(np.float32)
            # Also store avg fMRI (N, V) for per-ROI soft target
            fmri_avg = raw_fmri.mean(axis=1).astype(np.float32)  # (N, V)
            self.fmri_avg = fmri_avg
            # Map each trial back to its original image index
            self.img_indices = np.repeat(np.arange(N), R)
            self.has_avg = True
            print(f"  Single-trial mode: ({N}, {R}, {V}) → ({N*R}, {V})")
            print(f"  Also loaded avg fMRI: {fmri_avg.shape}")
        elif raw_fmri.ndim == 3:
            fmri = raw_fmri.mean(axis=1).astype(np.float32)
            self.fmri_avg = fmri   # avg IS the fmri in this case
            self.img_indices = None
            self.has_avg = False
        elif raw_fmri.ndim == 2:
            fmri = raw_fmri.astype(np.float32)
            self.fmri_avg = fmri
            self.img_indices = None
            self.has_avg = False
        else:
            raise ValueError(f"Unexpected fMRI shape: {raw_fmri.shape}")
        del raw_fmri
        self.fmri = fmri

        # Determine image loading mode
        self.image_dir = image_dir
        self.stim_mmap = None
        self.use_png = False

        if image_dir and os.path.isdir(image_dir):
            # Use individual PNG files
            self.use_png = True
            print(f"  Images: PNG from {image_dir}")
        elif stim_path and os.path.exists(stim_path):
            # Use numpy stim array
            self.stim_mmap = np.load(stim_path, mmap_mode='r')
            print(f"  Stim: {self.stim_mmap.shape} {self.stim_mmap.dtype} "
                  f"(mmap)")
        else:
            raise FileNotFoundError(
                f"No image source found. Tried dir={image_dir}, "
                f"stim={stim_path}")

        self.n_samples = fmri.shape[0]
        if max_samples > 0:
            self.n_samples = min(max_samples, self.n_samples)
            self.fmri = self.fmri[:self.n_samples]
            if self.img_indices is not None:
                self.img_indices = self.img_indices[:self.n_samples]
            # Note: fmri_avg keeps full N entries; we use img_indices to index it

        # Image transforms (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

        print(f"  fMRI: {self.fmri.shape}")
        print(f"  {split}: {self.n_samples} samples")
        if max_samples > 0:
            print(f"  Debug: limited to {max_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).float()

        # Get original image index (for single-trial mode)
        img_idx = int(self.img_indices[idx]) if self.img_indices is not None \
            else idx

        # Average fMRI (same image, averaged over 3 trials)
        fmri_avg = torch.from_numpy(self.fmri_avg[img_idx]).float()

        # Load image
        if self.use_png:
            img_path = os.path.join(self.image_dir, f"{img_idx}.png")
            image = Image.open(img_path).convert('RGB')
        else:
            # From numpy stim array (H, W, C) uint8
            img_array = np.array(self.stim_mmap[img_idx])
            image = Image.fromarray(img_array)

        image = self.transform(image)
        return fmri, fmri_avg, image


# ─── Utilities ────────────────────────────────────────────────────────────────


def pearson_corr_voxelwise(pred, target):
    """Per-voxel Pearson correlation averaged across voxels."""
    pred_zm = pred - pred.mean(0, keepdim=True)
    tgt_zm = target - target.mean(0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()


def pearson_corr_samplewise(pred, target):
    """Per-sample Pearson correlation averaged across samples."""
    pred_zm = pred - pred.mean(1, keepdim=True)
    tgt_zm = target - target.mean(1, keepdim=True)
    num = (pred_zm * tgt_zm).sum(1)
    den = (pred_zm.norm(dim=1) * tgt_zm.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


def ema_update(source, target, decay):
    with torch.no_grad():
        for s, t in zip(source.parameters(), target.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)
        # Also copy buffers (e.g. BatchNorm running stats)
        for s, t in zip(source.buffers(), target.buffers()):
            t.data.copy_(s.data)


def cosine_lr(optimizer, epoch, total, warmup, base_lrs, min_lr=1e-6):
    """Cosine LR scheduler supporting per-group base LRs."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs] * len(optimizer.param_groups)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if epoch < warmup:
            lr = base_lr * epoch / max(warmup, 1)
        else:
            p = (epoch - warmup) / max(total - warmup, 1)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * p))
        pg["lr"] = lr
    return [pg["lr"] for pg in optimizer.param_groups]


# ─── Velocity Loss Functions ─────────────────────────────────────────────────


def compute_velocity_loss(v_pred, ut, t=None, loss_type="mse",
                          huber_c=0.1,
                          timestep_weight_type="none",
                          direction_weight=1.0,
                          magnitude_weight=0.1):
    """
    Compute velocity matching loss.

    Args:
        v_pred: (B, D) predicted velocity
        ut: (B, D) target velocity
        t: (B,) timestep values for weighting
        loss_type: "mse" | "pseudo_huber" | "decoupled"
    """
    info = {}

    if loss_type == "pseudo_huber":
        diff_sq = (v_pred - ut).pow(2)
        per_element = (diff_sq + huber_c ** 2).sqrt() - huber_c
        raw_loss = per_element.mean(dim=-1)
        info["huber_raw"] = raw_loss.mean().item()

    elif loss_type == "decoupled":
        cos_sim = F.cosine_similarity(v_pred, ut, dim=-1)
        dir_loss = 1.0 - cos_sim
        pred_mag = v_pred.norm(dim=-1).clamp(min=1e-6)
        tgt_mag = ut.norm(dim=-1).clamp(min=1e-6)
        mag_loss = (pred_mag.log() - tgt_mag.log()).pow(2)
        raw_loss = direction_weight * dir_loss + magnitude_weight * mag_loss
        info["dir_loss"] = dir_loss.mean().item()
        info["mag_loss"] = mag_loss.mean().item()
        info["v_cos"] = cos_sim.mean().item()

    else:  # "mse"
        raw_loss = (v_pred - ut).pow(2).mean(dim=-1)

    # Timestep weighting
    if t is not None and timestep_weight_type != "none":
        if timestep_weight_type == "linear":
            w = 1.0 - t
        elif timestep_weight_type == "cosine":
            w = torch.cos(t * 3.14159 / 2).pow(2)
        elif timestep_weight_type == "snr":
            w = 1.0 / (1.0 + t / (1.0 - t + 1e-4))
        else:
            w = torch.ones_like(t)
        w = w / (w.mean() + 1e-8)
        loss = (w * raw_loss).mean()
        info["w_mean"] = w.mean().item()
    else:
        loss = raw_loss.mean()

    return loss, info


# ─── ODE Wrapper ─────────────────────────────────────────────────────────────


class BrainFlowV2ODEWrapper(torch.nn.Module):
    """ODE wrapper for BrainFlowV2: noise → fMRI voxels."""

    def __init__(self, model, image, cfg_scale=1.0):
        super().__init__()
        self.model = model
        self.image = image
        self.cfg_scale = cfg_scale

    def forward(self, t, x):
        B = x.shape[0]
        t_batch = t.expand(B)
        if self.cfg_scale == 1.0:
            return self.model(t_batch, x, self.image)
        else:
            return self.model.forward_with_cfg(
                t_batch, x, self.image, self.cfg_scale)


@torch.no_grad()
def stochastic_euler_sample(model, x0, image, T, cfg_scale=1.0):
    """FlowNP-style stochastic Euler sampling."""
    z = x0
    B = z.shape[0]
    for i in range(T):
        tt = i / T
        t_batch = torch.full((B,), tt, device=z.device)
        if cfg_scale == 1.0:
            v_pred = model(t_batch, z, image)
        else:
            v_pred = model.forward_with_cfg(
                t_batch, z, image, cfg_scale)
        alpha = 1 + tt * (1 - tt)
        sigma = 0.2 * (tt * (1 - tt)) ** 0.5
        z = z + (alpha * v_pred + sigma * torch.randn_like(z)) / T
    return z


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, val_loader, fm, device, ode_steps=50,
             cfg_scale=1.0, num_trials=1, decomposer=None,
             sampler="ode"):
    """Validate BrainFlowV2 — generate fMRI directly from images."""
    if sampler == "ode":
        from torchdiffeq import odeint

    model.eval()
    total_flow_loss = 0
    n_batches = 0
    all_pred, all_true = [], []
    all_v_cos = []

    for fmri, fmri_avg, image in val_loader:
        fmri, image = fmri.to(device), image.to(device)
        # fmri_avg not needed in val (val always uses averaged fMRI)

        # Flow loss
        x0 = torch.randn_like(fmri)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, fmri)
        v_pred = model(t, xt, image)

        flow_loss = F.mse_loss(v_pred, ut)
        total_flow_loss += flow_loss.item()
        n_batches += 1

        cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
        all_v_cos.append(cos)

        # Generation: noise → fMRI directly
        fmri_gen = torch.zeros_like(fmri)
        for _ in range(num_trials):
            x0_trial = torch.randn_like(fmri)
            if sampler == "stochastic_euler":
                fmri_trial = stochastic_euler_sample(
                    model, x0_trial, image, T=ode_steps,
                    cfg_scale=cfg_scale)
            else:
                ode_fn = BrainFlowV2ODEWrapper(model, image, cfg_scale)
                t_span = torch.linspace(0, 1, ode_steps, device=device)
                traj = odeint(ode_fn, x0_trial, t_span, method="midpoint")
                fmri_trial = traj[-1]
            fmri_gen += fmri_trial
        fmri_gen = fmri_gen / num_trials

        all_pred.append(fmri_gen)
        all_true.append(fmri)

    model.train()

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)

    metrics = {
        "val_flow_loss": total_flow_loss / max(n_batches, 1),
        "val_v_cos": sum(all_v_cos) / len(all_v_cos),
        "val_fmri_mse": F.mse_loss(preds, trues).item(),
        "val_fmri_pcc": pearson_corr_voxelwise(preds, trues),
        "val_fmri_spcc": pearson_corr_samplewise(preds, trues),
    }

    # Per-ROI metrics
    if decomposer is not None:
        for roi in decomposer.rois:
            if roi.n_voxels > 10:
                p = preds[:, roi.indices]
                t_roi = trues[:, roi.indices]
                metrics[f"roi_{roi.name}_spcc"] = (
                    pearson_corr_samplewise(p, t_roi))
            else:
                metrics[f"roi_{roi.name}_spcc"] = 0.0

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        "BrainFlow V2 — FlowFM-Style Direct Flow Matching")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest.pt in output_dir")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to a specific checkpoint .pt file to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    num_epochs = 2 if args.debug else train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    encoder_lr = train_cfg.get("encoder_lr", lr * 0.1)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    use_ema = train_cfg.get("use_ema", True)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    cfg_drop_prob = train_cfg.get("cfg_drop_prob", 0.5)
    cfg_scale = train_cfg.get("cfg_scale", 1.0)
    ode_steps = train_cfg.get("ode_steps", 50)
    num_trials = train_cfg.get("num_trials", 1)
    eval_interval = 1 if args.debug else train_cfg.get("eval_interval", 5)
    sampler = train_cfg.get("sampler", "ode")
    freeze_encoder_epochs = train_cfg.get("freeze_encoder_epochs", 0)

    # Per-ROI soft target options
    use_per_roi_soft = train_cfg.get("use_per_roi_soft_target", True)
    # alpha per ROI: alpha=1 → use only avg, alpha=0 → use only single trial
    # Default: scaled by ROI hierarchy (early=low alpha, late/other=high alpha)
    _default_roi_alphas = {
        "V1":    0.3,   # high SNR retinotopic, keep more trial variability
        "V2":    0.3,
        "V3":    0.5,
        "hV4":   0.5,
        "body":  0.65,  # floc areas more variable
        "face":  0.65,
        "place": 0.65,
        "word":  0.65,
        "other": 0.8,   # mixed voxels, mostly use avg
    }
    _cfg_roi_alphas = train_cfg.get("roi_soft_alphas", {})
    # Merge defaults with any user overrides from config
    roi_alpha_map = {**_default_roi_alphas, **_cfg_roi_alphas}

    # Flow options
    use_ot = train_cfg.get("use_ot", True)
    sigma = train_cfg.get("sigma", 0.0)

    timestep_sampling = train_cfg.get("timestep_sampling", "logit_normal")
    logit_normal_mu = train_cfg.get("logit_normal_mu", 0.0)
    logit_normal_sigma = train_cfg.get("logit_normal_sigma", 1.0)

    # Loss options
    velocity_loss_type = train_cfg.get("velocity_loss_type", "mse")
    huber_c = train_cfg.get("huber_c", 0.1)
    timestep_weight_type = train_cfg.get("timestep_weight_type", "none")
    direction_weight = train_cfg.get("direction_weight", 1.0)
    magnitude_weight = train_cfg.get("magnitude_weight", 0.1)

    image_size = data_cfg.get("image_size", 224)
    use_single_trials = data_cfg.get("use_single_trials", False)

    output_dir = cfg.get("output_dir", "results/brainflow_v2")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Logger ──
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, mode='w')],
    )
    logger = logging.getLogger('brainflow_v2')
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # ── ROI Decomposer ──
    roi_dir = data_cfg.get(
        "roi_dir", "Data/nsddata/ppdata/subj01/func1pt8mm/roi")
    decomposer = ROIDecomposer(roi_dir)
    roi_names = decomposer.get_roi_names()
    logger.info(f"\n{decomposer.summary()}")

    # ── ROI-Sort Permutation (for ROI-aware patching) ──
    # Build once from ROIDecomposer; inject into model config automatically.
    # Set use_roi_pos_embed: false in config.yaml to disable.
    if model_cfg.get("use_roi_pos_embed", True):
        patch_size = model_cfg.get("patch_size", 124)
        n_voxels = model_cfg.get("n_voxels", 15724)
        voxel_perm, patch_roi_ids = build_roi_perm(
            decomposer, patch_size, n_voxels)
        model_cfg["voxel_perm"] = voxel_perm.tolist()
        model_cfg["patch_roi_ids"] = patch_roi_ids.tolist()
        model_cfg["use_roi_pos_embed"] = True
        logger.info(
            f"ROI-sort: built voxel_perm ({len(voxel_perm)} voxels), "
            f"patch_roi_ids ({len(patch_roi_ids)} patches)")
    else:
        logger.info("ROI-sort: disabled (use_roi_pos_embed=false)")

    # ── Per-ROI soft alpha list (ordered by decomposer.rois) ──
    # roi_soft_alphas[i] = alpha for decomposer.rois[i]
    roi_soft_alphas = [roi_alpha_map.get(roi.name, 0.5)
                       for roi in decomposer.rois]
    if use_per_roi_soft:
        alpha_str = ", ".join(
            f"{roi.name}={a:.2f}" for roi, a in
            zip(decomposer.rois, roi_soft_alphas))
        logger.info(f"Per-ROI soft target: ON | {alpha_str}")
    else:
        logger.info("Per-ROI soft target: OFF (use_per_roi_soft_target=false)")

    # ── Data ──
    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    root = data_cfg["root"]

    debug_n = 128 if args.debug else 0

    # Build paths
    train_fmri = os.path.join(
        root, subject, f"nsd_train_fmri_zscore_sub{sub_num}.npy")
    train_stim = os.path.join(
        root, subject, f"nsd_train_stim_sub{sub_num}.npy")
    train_img_dir = os.path.join(root, subject, "train_img")

    val_fmri = os.path.join(
        root, subject, f"nsd_test_fmri_zscore_sub{sub_num}.npy")
    val_stim = os.path.join(
        root, subject, f"nsd_test_stim_sub{sub_num}.npy")
    val_img_dir = os.path.join(root, subject, "test_img")

    train_ds = FmriImageDataset(
        train_fmri, train_stim, train_img_dir,
        split="train", image_size=image_size, max_samples=debug_n,
        use_single_trials=use_single_trials)
    val_ds = FmriImageDataset(
        val_fmri, val_stim, val_img_dir,
        split="test", image_size=image_size,
        max_samples=debug_n // 4 if args.debug else 0,
        use_single_trials=False)  # val always averaged for fair comparison

    if args.debug:
        batch_size = min(batch_size, 16)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        drop_last=(not args.debug))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Model ──
    model = BrainFlowV2(BrainFlowV2Config(**model_cfg)).to(device)
    ema_model = copy.deepcopy(model) if use_ema else None
    pc = model.param_count()
    logger.info(
        f"BrainFlowV2: encoder={pc['encoder_M']:.1f}M "
        f"cond={pc['cond_M']:.2f}M ctx={pc['ctx_proj_M']:.2f}M "
        f"blocks={pc['blocks_M']:.1f}M "
        f"embed={pc['embed_M']:.2f}M output={pc['output_M']:.3f}M "
        f"total={pc['total_M']:.1f}M | EMA={'ON' if use_ema else 'OFF'}")

    # ── Flow Matcher ──
    if use_ot:
        try:
            import ot
            fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
            logger.info(f"Using ExactOT Flow Matcher (sigma={sigma})")
        except ImportError:
            logger.warning("POT not found, reverting to standard CFM")
            fm = ConditionalFlowMatcher(sigma=sigma)
            use_ot = False
    else:
        fm = ConditionalFlowMatcher(sigma=sigma)
        logger.info(f"Using standard CFM (sigma={sigma})")

    # ── Optimizer (separate LR for encoder) ──
    optimizer = torch.optim.AdamW([
        {"params": model.get_encoder_params(),
         "lr": encoder_lr, "name": "encoder"},
        {"params": model.get_velocity_params(),
         "lr": lr, "name": "velocity"},
    ], weight_decay=train_cfg.get("weight_decay", 0.05))
    base_lrs = [encoder_lr, lr]

    # ── Resume from checkpoint ──
    start_epoch = 1
    best_pcc = -1.0
    patience_counter = 0

    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = os.path.join(output_dir, "latest.pt")

    if resume_path and os.path.exists(resume_path):
        logger.info(f"\n► Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_ema and "ema_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_pcc = ckpt.get("best_pcc", -1.0)
        logger.info(
            f"  Resumed epoch={ckpt['epoch']} | best_pcc={best_pcc:.4f} "
            f"| continuing from epoch {start_epoch}")
    elif resume_path:
        logger.warning(f"  Checkpoint not found at {resume_path}, training from scratch")
    logger.info(
        f"Optimizer: encoder_lr={encoder_lr:.2e}, velocity_lr={lr:.2e}")

    # ── History CSV ──
    history_path = os.path.join(output_dir, "history.csv")
    roi_fields = [f"roi_{n}_spcc" for n in roi_names]
    fields = [
        "epoch", "train_loss", "lr_enc", "lr_vel",
        "grad_avg", "grad_max",
        "val_flow_loss", "val_v_cos",
        "val_fmri_mse", "val_fmri_pcc", "val_fmri_spcc",
    ] + roi_fields
    # On resume: append to existing CSV. On fresh start: write header.
    is_resuming = (start_epoch > 1)
    with open(history_path, "a" if is_resuming else "w", newline="") as f:
        if not is_resuming:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    best_pcc = best_pcc  # already set above (from checkpoint or default -1.0)
    patience_counter = patience_counter  # already set above
    patience = train_cfg.get("patience", 200)

    # ── Training ──
    ts_info = f"timestep_sampling={timestep_sampling}"
    if timestep_sampling == "logit_normal":
        ts_info += f" (mu={logit_normal_mu}, sigma={logit_normal_sigma})"
    logger.info(
        f"Training {num_epochs} epochs, eval every {eval_interval} "
        f"| {ts_info}")
    logger.info(
        f"BRAINFLOW V2 (FlowFM) | OT={use_ot} "
        f"CFG_drop={cfg_drop_prob} Sampler={sampler} "
        f"freeze_enc={freeze_encoder_epochs}ep")
    logger.info(
        f"LOSS: type={velocity_loss_type} | "
        f"t_weight={timestep_weight_type}"
        + (f" | huber_c={huber_c}"
           if velocity_loss_type == "pseudo_huber" else ""))

    # Precompute ROI index tensors on device (once) for per-ROI soft target
    # Avoids creating a new tensor every batch inside the training loop
    if use_per_roi_soft:
        roi_index_tensors = [
            torch.tensor(roi.indices, dtype=torch.long, device=device)
            for roi in decomposer.rois
        ]
    else:
        roi_index_tensors = None

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()

        # Encoder freeze/unfreeze
        if freeze_encoder_epochs > 0:
            if epoch <= freeze_encoder_epochs:
                if epoch == 1:
                    model.freeze_encoder()
                    logger.info(
                        f"  Encoder FROZEN for {freeze_encoder_epochs} epochs")
            elif epoch == freeze_encoder_epochs + 1:
                model.unfreeze_encoder()
                logger.info("  Encoder UNFROZEN")

        current_lrs = cosine_lr(
            optimizer, epoch - 1, num_epochs, warmup_epochs, base_lrs)

        ep_total, n_steps = 0, 0
        grads_all, grads_max_all = [], []
        t0 = time.time()

        for batch_idx, (fmri, fmri_avg, image) in enumerate(train_loader):
            fmri, fmri_avg, image = (
                fmri.to(device), fmri_avg.to(device), image.to(device))
            B = fmri.shape[0]

            # ─── Per-ROI Soft Target ───────────────────────────────────────
            # Build x1 as weighted mix of single trial + avg per ROI.
            # ROIs with high SNR (V1/V2) keep more single-trial variability;
            # noisy ROIs (other) use mostly avg to reduce velocity conflict.
            if use_per_roi_soft:
                x1 = fmri.clone()
                for roi_indices, alpha in zip(roi_index_tensors, roi_soft_alphas):
                    # x1_r = alpha * avg_r + (1-alpha) * trial_r
                    x1[:, roi_indices] = (
                        alpha * fmri_avg[:, roi_indices]
                        + (1 - alpha) * fmri[:, roi_indices]
                    )
            else:
                x1 = fmri
            # ──────────────────────────────────────────────────────────────

            # ─── Flow Matching: x0=noise, x1=soft_target ──────────────────
            x0 = torch.randn_like(x1)

            # DGS: Dynamic Guidance Switching on representation
            drop_mask = torch.rand(B, device=device) < cfg_drop_prob

            # Timestep sampling
            if timestep_sampling == "logit_normal":
                u = torch.randn(B, device=device)
                t_sample = torch.sigmoid(
                    logit_normal_mu + logit_normal_sigma * u)
                t_expand = t_sample[:, None]
                xt = t_expand * x1 + (1 - t_expand) * x0
                ut = x1 - x0
                t = t_sample
            else:
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)

            # Forward: predict velocity
            # For DGS, we need per-sample drop
            # First, get representation for all
            cls_token, patch_tokens = model.encoder(image)
            # cls_token: (B, enc_dim), patch_tokens: (B, N, enc_dim)
            if drop_mask.any():
                cls_token = cls_token.clone()
                patch_tokens = patch_tokens.clone()
                cls_token[drop_mask] = 0.0
                patch_tokens[drop_mask] = 0.0

            # Manually compute forward with pre-computed representations
            t_emb = timestep_embedding(
                t * 1000, model.config.hidden_dim).to(device)
            c = model.cond_mlp(cls_token, t_emb)

            # Context for cross-attention
            context = None
            if model.use_cross_attention:
                context = model.context_norm(
                    model.context_proj(patch_tokens))

            x_patches = model._patchify(xt)
            x_tokens = model.patch_embed(x_patches)
            # Positional embedding: ROI-aware or legacy
            if model.use_roi_pos_embed and model.roi_embed is not None:
                roi_pos = model.roi_embed(model.patch_roi_ids).unsqueeze(0)
                x_tokens = x_tokens + roi_pos + model.patch_pos_bias
            else:
                x_tokens = x_tokens + model.patch_pos_embed

            for block in model.blocks:
                x_tokens = block(x_tokens, c, context)

            mod_params = model.final_adaLN(c)
            shift, scale = mod_params.chunk(2, dim=-1)
            x_out = modulate(
                model.final_layer_norm(x_tokens), shift, scale)
            x_out = model.output_proj(x_out)
            v_pred = model._unpatchify(x_out)

            # Loss
            loss, loss_info = compute_velocity_loss(
                v_pred, ut, t=t,
                loss_type=velocity_loss_type,
                huber_c=huber_c,
                timestep_weight_type=timestep_weight_type,
                direction_weight=direction_weight,
                magnitude_weight=magnitude_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip if grad_clip > 0 else float('inf'))
            optimizer.step()
            if use_ema:
                ema_update(model, ema_model, ema_decay)

            ep_total += loss.item()
            grads_all.append(gn.item())
            grads_max_all.append(gn.item())
            n_steps += 1

            if batch_idx == 0 and epoch <= 5:
                with torch.no_grad():
                    v_cos = F.cosine_similarity(
                        v_pred, ut, dim=-1).mean().item()
                logger.info(
                    f"  [Ep{epoch} B0] loss={loss.item():.4f} "
                    f"v_cos={v_cos:.4f}")

        avg_total = ep_total / max(n_steps, 1)
        avg_grad = sum(grads_all) / len(grads_all)
        max_grad = max(grads_max_all)
        ep_time = time.time() - t0

        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) "
            f"[BRAINFLOW_V2] | loss={avg_total:.5f} | "
            f"lr_enc={current_lrs[0]:.2e} lr_vel={current_lrs[1]:.2e} "
            f"grad={avg_grad:.4f}")

        # ── Eval ──
        if epoch % eval_interval == 0 or epoch == 1:
            eval_model = ema_model if use_ema else model
            val = validate(eval_model, val_loader, fm, device,
                           ode_steps, cfg_scale, num_trials,
                           decomposer=decomposer, sampler=sampler)

            row = {
                "epoch": epoch,
                "train_loss": f"{avg_total:.6f}",
                "lr_enc": f"{current_lrs[0]:.2e}",
                "lr_vel": f"{current_lrs[1]:.2e}",
                "grad_avg": f"{avg_grad:.4f}",
                "grad_max": f"{max_grad:.4f}",
                **{k: f"{v:.6f}" if (
                    'loss' in k or 'mse' in k or 'std' in k or
                    'ratio' in k) else f"{v:.4f}"
                   for k, v in val.items()},
            }
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            spcc = val["val_fmri_spcc"]
            is_best = spcc > best_pcc
            logger.info(
                f"  VAL | v_cos={val['val_v_cos']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} "
                f"f_pcc={val['val_fmri_pcc']:.4f} "
                f"f_spcc={spcc:.4f}{'  ★' if is_best else ''}")

            # Per-ROI PCC
            roi_str = " | ".join(
                f"{n}={val.get(f'roi_{n}_spcc', 0):.3f}"
                for n in roi_names)
            logger.info(f"  ROI | {roi_str}")

            if is_best:
                best_pcc = spcc
                patience_counter = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pcc": best_pcc,
                    "config": cfg,
                }
                torch.save(save_dict,
                           os.path.join(output_dir, "best_model.pt"))
                logger.info(f"  ★ Saved best (PCC={best_pcc:.4f})")
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # Save latest.pt every epoch (enables resume at any point)
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_pcc": best_pcc,
            "config": cfg,
        }
        if use_ema:
            save_dict["ema_state_dict"] = ema_model.state_dict()
        torch.save(save_dict, os.path.join(output_dir, "latest.pt"))

        # Save periodic snapshot (every save_every epochs)
        if epoch % train_cfg.get("save_every", 50) == 0:
            torch.save(save_dict,
                       os.path.join(output_dir, f"ckpt_ep{epoch:04d}.pt"))

    logger.info(f"Done! Best PCC: {best_pcc:.4f}")


if __name__ == "__main__":
    main()
