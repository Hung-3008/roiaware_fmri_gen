"""
Stage 2: Cross-Attention Flow Matching with Self-Conditioning.

Architecture:  BrainXAttnFlowDiT (Cross-Attn DiT + Perceiver Bottleneck)
Flow:          N(0,I) → z_true, conditioned on DINOv2 multi-layer features
Training:      Self-conditioning (50% prob) + multi-timestep consistency
Loss:          Noise-robust velocity matching:
                 - σ-augmented targets (noisy targets prevent memorization)
                 - Huber loss (bounded gradient for noise outliers)
                 - Per-dim SNR weights (down-weight noisy latent dimensions)

Usage:
    python -m src.train_stage2_xattn_flow --config src/configs/subj01/stage2_xattn_flow_vit_vae.yaml
    python -m src.train_stage2_xattn_flow --config src/configs/subj01/stage2_xattn_flow_vit_vae.yaml --debug
"""

import argparse
import copy
import csv
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from src.model.brain_xattn_flow_dit import BrainXAttnFlowDiT, BrainXAttnFlowDiTConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig
from src.model.fmri_vit_vae import FmriViTVAE, create_fmri_vit_vae
from src.utils.roi_utils import ROIDecomposer


# ─── Dataset ──────────────────────────────────────────────────────────────────


class FmriMultiLayerDataset(Dataset):
    """Dataset pairing fMRI with multi-layer DINOv2 features."""

    def __init__(self, fmri_path, dino_path, split="train", max_samples=0):
        print(f"\nFmriMultiLayerDataset [{split}]: Loading...")
        raw_fmri = np.load(fmri_path)
        self.dino_mmap = np.load(dino_path, mmap_mode='r')

        if raw_fmri.ndim == 3:
            fmri = raw_fmri.mean(axis=1).astype(np.float32)
        elif raw_fmri.ndim == 2:
            fmri = raw_fmri.astype(np.float32)
        else:
            raise ValueError(f"Unexpected fMRI shape: {raw_fmri.shape}")
        del raw_fmri

        assert fmri.shape[0] == self.dino_mmap.shape[0], \
            f"Mismatch: fMRI {fmri.shape[0]} vs DINOv2 {self.dino_mmap.shape[0]}"

        self.fmri = fmri
        self.n_samples = fmri.shape[0]

        if max_samples > 0:
            self.fmri = self.fmri[:max_samples]
            self.n_samples = min(max_samples, self.n_samples)

        print(f"  DINOv2: {self.dino_mmap.shape} (multi-layer, mmap)")
        print(f"  {split}: {self.n_samples} samples")
        if max_samples > 0:
            print(f"  Debug: limited to {max_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).float()
        dino = torch.from_numpy(np.array(self.dino_mmap[idx])).float()
        return fmri, dino


# ─── Utilities ────────────────────────────────────────────────────────────────


def pearson_corr_voxelwise(pred, target):
    pred_zm = pred - pred.mean(0, keepdim=True)
    tgt_zm = target - target.mean(0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()


def pearson_corr_samplewise(pred, target):
    pred_zm = pred - pred.mean(1, keepdim=True)
    tgt_zm = target - target.mean(1, keepdim=True)
    num = (pred_zm * tgt_zm).sum(1)
    den = (pred_zm.norm(dim=1) * tgt_zm.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


def ema_update(source, target, decay):
    with torch.no_grad():
        for s, t in zip(source.parameters(), target.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)


def cosine_lr(optimizer, epoch, total, warmup, base_lr, min_lr=1e-6):
    if epoch < warmup:
        lr = base_lr * epoch / max(warmup, 1)
    else:
        p = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─── ODE Wrapper (with self-conditioning) ────────────────────────────────────


class SelfCondFlowODEWrapper(torch.nn.Module):
    """ODE wrapper for self-conditioned flow: noise → z_true.

    At each ODE step, first computes a rough estimate z_hat (no self-cond),
    then refines with self-conditioning for the final velocity.
    """

    def __init__(self, model, context, cfg_scale=1.0,
                 use_self_cond=True):
        super().__init__()
        self.model = model
        self.context = context
        self.cfg_scale = cfg_scale
        self.use_self_cond = use_self_cond
        self._z_prev_est = None  # running self-estimate from previous step

    def forward(self, t, z):
        B = z.shape[0]
        t_batch = t.expand(B)

        # Self-conditioning: use estimate from previous ODE step
        z_self_cond = self._z_prev_est

        if self.cfg_scale == 1.0:
            v = self.model.forward_flow(
                t_batch, z, self.context, z_self_cond=z_self_cond)
        else:
            v = self.model.forward_flow_with_cfg(
                t_batch, z, self.context, self.cfg_scale,
                z_self_cond=z_self_cond)

        # Update running estimate for next step: z_est = z_t + (1-t)*v
        if self.use_self_cond:
            with torch.no_grad():
                t_weight = (1 - t).view(-1, 1) if t.dim() == 0 else (1 - t)
                self._z_prev_est = (z + t_weight * v).detach()

        return v


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, vae, val_loader, fm, device, ode_steps=50,
             cfg_scale=1.0, num_trials=1, decomposer=None,
             use_self_cond=True):
    from torchdiffeq import odeint

    model.eval()
    total_flow_loss = 0
    n_batches = 0
    all_pred, all_true = [], []
    all_z_gen, all_z_true = [], []
    all_v_cos = []

    for fmri, dino in val_loader:
        fmri, dino = fmri.to(device), dino.to(device)
        z1, _, _ = vae.encode(fmri, sample_posterior=False)

        # Flow loss (no self-conditioning for fair comparison)
        x0 = torch.randn_like(z1)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, z1)
        v_pred = model.forward_flow(t, xt, dino, z_self_cond=None)
        flow_loss = F.mse_loss(v_pred, ut)
        total_flow_loss += flow_loss.item()
        n_batches += 1

        cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
        all_v_cos.append(cos)

        # ODE generation with self-conditioning
        ode_fn = SelfCondFlowODEWrapper(
            model, dino, cfg_scale, use_self_cond=use_self_cond)
        t_span = torch.linspace(0, 1, ode_steps, device=device)

        z_gen = torch.zeros_like(z1)
        for _ in range(num_trials):
            ode_fn._z_prev_est = None  # reset for each trial
            x0_trial = torch.randn_like(z1)
            traj = odeint(ode_fn, x0_trial, t_span, method="midpoint")
            z_gen = z_gen + traj[-1]
        z_gen = z_gen / num_trials

        fmri_pred = vae.decode(z_gen)

        all_z_gen.append(z_gen)
        all_z_true.append(z1)
        all_pred.append(fmri_pred)
        all_true.append(fmri)

    model.train()

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)
    z_gens = torch.cat(all_z_gen)
    z_trues = torch.cat(all_z_true)

    z_gen_std = z_gens.std().item()
    z_gen_cross_var = z_gens.var(dim=0).mean().item()
    z_true_cross_var = z_trues.var(dim=0).mean().item()

    metrics = {
        "val_flow_loss": total_flow_loss / max(n_batches, 1),
        "val_v_cos": sum(all_v_cos) / len(all_v_cos),
        "val_latent_mse": F.mse_loss(z_gens, z_trues).item(),
        "val_latent_pcc": pearson_corr_samplewise(z_gens, z_trues),
        "val_zgen_std": z_gen_std,
        "val_zgen_crossvar_ratio": z_gen_cross_var / max(
            z_true_cross_var, 1e-8),
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
                metrics[f"roi_{roi.name}_spcc"] = pearson_corr_samplewise(
                    p, t_roi)
            else:
                metrics[f"roi_{roi.name}_spcc"] = 0.0

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        "Stage 2: Cross-Attention Flow + Self-Conditioning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
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
    grad_clip = train_cfg.get("grad_clip", 1.0)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    use_ema = train_cfg.get("use_ema", True)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    cfg_drop_prob = train_cfg.get("cfg_drop_prob", 0.15)
    cfg_scale = train_cfg.get("cfg_scale", 1.5)
    ode_steps = train_cfg.get("ode_steps", 50)
    num_trials = train_cfg.get("num_trials", 1)
    eval_interval = 1 if args.debug else train_cfg.get("eval_interval", 5)
    flow_weight = train_cfg.get("flow_weight", 1.0)
    self_cond_prob = train_cfg.get("self_cond_prob", 0.5)
    consistency_weight = train_cfg.get("consistency_weight", 0.5)

    # Noise-robust loss params
    sigma_aug = train_cfg.get("sigma_aug", 0.3)
    huber_delta = train_cfg.get("huber_delta", 1.0)
    use_snr_weights = train_cfg.get("use_snr_weights", True)

    output_dir = cfg.get("output_dir", "results/stage2_xattn_flow")
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
    logger = logging.getLogger('stage2_xattn_flow')
    logger.info(f"Config: {cfg}")

    # ── ROI Decomposer ──
    roi_dir = data_cfg.get("roi_dir",
                           "Data/nsddata/ppdata/subj01/func1pt8mm/roi")
    decomposer = ROIDecomposer(roi_dir)
    roi_names = decomposer.get_roi_names()
    logger.info(f"\n{decomposer.summary()}")

    # ── Data ──
    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    root = data_cfg["root"]

    dino_suffix = data_cfg.get("dino_suffix", "dinov2_vitl14_multilayer")
    debug_n = 128 if args.debug else 0
    train_ds = FmriMultiLayerDataset(
        os.path.join(root, subject,
                     f"nsd_train_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject,
                     f"nsd_{dino_suffix}_train_sub{sub_num}.npy"),
        split="train", max_samples=debug_n)
    val_ds = FmriMultiLayerDataset(
        os.path.join(root, subject,
                     f"nsd_test_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject,
                     f"nsd_{dino_suffix}_test_sub{sub_num}.npy"),
        split="test", max_samples=debug_n // 4 if args.debug else 0)

    if args.debug:
        batch_size = min(batch_size, 32)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        drop_last=(not args.debug))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Frozen VAE ──
    vae_ckpt = data_cfg["vae_checkpoint"]
    config_path = os.path.join(os.path.dirname(vae_ckpt), "config.yaml")
    if not os.path.exists(config_path):
        subject = data_cfg.get("subject", "subj01")
        for alt in [
            f"src/configs/{subject}/stage1_vit_vae.yaml",
            f"src/configs/exp/fmri_mlp_vae_768_{subject}.yaml",
        ]:
            if os.path.exists(alt):
                config_path = alt
                break
        logger.info(f"VAE config not found at ckpt dir, using {config_path}")

    with open(config_path) as f:
        vae_cfg = yaml.safe_load(f)
    vae_model_type = vae_cfg.get("model_type", "mlp")
    if vae_model_type == "vit":
        vae = create_fmri_vit_vae(**vae_cfg["model"]).to(device).eval()
        logger.info("VAE type: ViT")
    else:
        vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
        logger.info("VAE type: MLP")
    ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    for p in vae.parameters():
        p.requires_grad = False
    logger.info(f"VAE loaded from {vae_ckpt}")

    # ── Compute per-dim SNR weights from VAE posterior ──
    snr_weights = None
    if use_snr_weights:
        logger.info("Computing per-dim SNR weights from VAE posterior...")
        all_logvar = []
        with torch.no_grad():
            for fmri_batch, _ in train_loader:
                _, _, logvar = vae.encode(
                    fmri_batch.to(device), sample_posterior=False)
                all_logvar.append(logvar)
        avg_logvar = torch.cat(all_logvar).mean(0)  # (latent_dim,)
        snr_weights = 1.0 / (1.0 + torch.exp(avg_logvar))  # high var → low weight
        snr_weights = snr_weights / snr_weights.mean()  # normalize to mean=1
        snr_weights = snr_weights.to(device)
        logger.info(
            f"SNR weights: min={snr_weights.min():.3f} "
            f"max={snr_weights.max():.3f} "
            f"std={snr_weights.std():.3f}")

    # ── Model ──
    model = BrainXAttnFlowDiT(BrainXAttnFlowDiTConfig(**model_cfg)).to(device)
    ema_model = copy.deepcopy(model) if use_ema else None
    pc = model.param_count()
    logger.info(
        f"BrainXAttnFlowDiT: flow={pc['flow_M']:.1f}M "
        f"bottleneck={pc['bottleneck_M']:.1f}M "
        f"total={pc['total_M']:.1f}M"
        f" | EMA={'ON' if use_ema else 'OFF'}"
        f" | SelfCond={'ON' if model_cfg.get('use_self_cond', True) else 'OFF'}")

    # ── Flow Matcher ──
    sigma = train_cfg.get("sigma", 0.0)
    fm = ConditionalFlowMatcher(sigma=sigma)
    logger.info(f"Using standard Conditional Flow Matcher (sigma={sigma})")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.05))

    # ── History ──
    history_path = os.path.join(output_dir, "history.csv")
    roi_fields = [f"roi_{n}_spcc" for n in roi_names]
    fields = [
        "epoch", "train_loss", "flow_loss", "consist_loss",
        "lr", "grad_avg", "grad_max",
        "val_flow_loss", "val_v_cos",
        "val_latent_mse", "val_latent_pcc",
        "val_zgen_std", "val_zgen_crossvar_ratio",
        "val_fmri_mse", "val_fmri_pcc", "val_fmri_spcc",
    ] + roi_fields
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_pcc = -1.0
    patience_counter = 0
    patience = train_cfg.get("patience", 200)

    # ── Layer mixing weight log ──
    mixing_log_path = os.path.join(output_dir, "layer_mixing.csv")
    dino_layer_names = cfg.get("dino_layers", [3, 6, 9, 12])
    mixing_fields = ["epoch"]
    for b in range(model_cfg.get("depth", 6)):
        for layer_id in dino_layer_names:
            mixing_fields.append(f"flow_block{b}_layer{layer_id}")
    with open(mixing_log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=mixing_fields).writeheader()

    # ── Training ──
    use_self_cond = model_cfg.get("use_self_cond", True)
    logger.info(
        f"Training {num_epochs} epochs, eval every {eval_interval}")
    logger.info(
        f"XATTN FLOW + SELF-COND | flow_weight={flow_weight} "
        f"consist_weight={consistency_weight} | "
        f"self_cond_prob={self_cond_prob} | "
        f"cfg_drop={cfg_drop_prob}")
    logger.info(
        f"NOISE-ROBUST LOSS | sigma_aug={sigma_aug} "
        f"huber_delta={huber_delta} "
        f"snr_weights={'ON' if use_snr_weights else 'OFF'}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(
            optimizer, epoch - 1, num_epochs, warmup_epochs, lr)

        ep_flow, ep_consist, ep_total, n_steps = 0, 0, 0, 0
        grads_all, grads_max_all = [], []
        t0 = time.time()

        for batch_idx, (fmri, dino) in enumerate(train_loader):
            fmri, dino = fmri.to(device), dino.to(device)
            B = fmri.shape[0]

            with torch.no_grad():
                z1, _, _ = vae.encode(fmri, sample_posterior=False)

            # ─── σ-Augmented Targets ─────────────────────────────
            # Add noise to z_true to prevent memorizing noise patterns
            if sigma_aug > 0:
                z1_aug = z1 + sigma_aug * torch.randn_like(z1)
            else:
                z1_aug = z1

            # ─── Standard Flow: x0 = noise, x1 = z1_aug ─────────
            x0 = torch.randn_like(z1_aug)

            # CFG dropout: zero out context for unconditional training
            context = dino
            if cfg_drop_prob > 0:
                drop = torch.rand(B, device=device) < cfg_drop_prob
                if drop.any():
                    context = context.clone()
                    context[drop] = 0.0

            # Timestep sampling (uniform) — using augmented target
            t, xt, ut = fm.sample_location_and_conditional_flow(
                x0, z1_aug)

            # ─── Self-conditioning ───────────────────────────────
            z_self_cond = None
            if use_self_cond and random.random() < self_cond_prob:
                with torch.no_grad():
                    v_first = model.forward_flow(
                        t, xt, context, z_self_cond=None)
                    z_self_cond = (
                        xt + (1 - t)[:, None] * v_first).detach()

            # Forward: predict velocity (with or without self-cond)
            v_pred = model.forward_flow(
                t, xt, context, z_self_cond=z_self_cond)

            # ─── Loss 1: Noise-Robust Velocity Matching ──────────
            # Apply per-dim SNR weighting + Huber loss
            diff = v_pred - ut
            if snr_weights is not None:
                diff = diff * snr_weights.sqrt()  # weight per dimension
            loss_flow = F.smooth_l1_loss(
                diff, torch.zeros_like(diff), beta=huber_delta)

            # ─── Loss 2: Multi-timestep Consistency ──────────────
            loss_consist = torch.tensor(0.0, device=device)
            if consistency_weight > 0:
                with torch.no_grad():
                    t2 = torch.rand(B, device=device).clamp(0.01, 0.99)
                    xt2 = t2[:, None] * z1_aug + (1 - t2[:, None]) * x0
                    ut2 = z1_aug - x0

                v_pred2 = model.forward_flow(
                    t2, xt2, context, z_self_cond=None)
                diff2 = v_pred2 - ut2
                if snr_weights is not None:
                    diff2 = diff2 * snr_weights.sqrt()
                loss_consist = F.smooth_l1_loss(
                    diff2, torch.zeros_like(diff2), beta=huber_delta)

            # ─── Combined loss ───────────────────────────────────
            loss = (flow_weight * loss_flow
                    + consistency_weight * loss_consist)

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip if grad_clip > 0 else float('inf'))
            optimizer.step()
            if use_ema:
                ema_update(model, ema_model, ema_decay)

            ep_flow += loss_flow.item()
            ep_consist += loss_consist.item()
            ep_total += loss.item()
            grads_all.append(gn.item())
            grads_max_all.append(gn.item())
            n_steps += 1

            if batch_idx == 0 and epoch <= 5:
                with torch.no_grad():
                    v_cos = F.cosine_similarity(
                        v_pred, ut, dim=-1).mean().item()
                sc_str = "SC" if z_self_cond is not None else "noSC"
                logger.info(
                    f"  [Ep{epoch} B0 {sc_str}] flow={loss_flow.item():.4f} "
                    f"consist={loss_consist.item():.4f} "
                    f"v_cos={v_cos:.4f}")

        avg_total = ep_total / max(n_steps, 1)
        avg_flow = ep_flow / max(n_steps, 1)
        avg_consist = ep_consist / max(n_steps, 1)
        avg_grad = sum(grads_all) / len(grads_all)
        max_grad = max(grads_max_all)
        ep_time = time.time() - t0

        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) [XATTN+SC] | "
            f"total={avg_total:.5f} flow={avg_flow:.5f} "
            f"consist={avg_consist:.5f} | "
            f"lr={current_lr:.2e} grad={avg_grad:.4f}")

        # ── Eval ──
        if epoch % eval_interval == 0 or epoch == 1:
            eval_model = ema_model if use_ema else model
            val = validate(eval_model, vae, val_loader, fm, device,
                           ode_steps, cfg_scale, num_trials,
                           decomposer=decomposer,
                           use_self_cond=use_self_cond)

            row = {"epoch": epoch,
                   "train_loss": f"{avg_total:.6f}",
                   "flow_loss": f"{avg_flow:.6f}",
                   "consist_loss": f"{avg_consist:.6f}",
                   "lr": f"{current_lr:.2e}",
                   "grad_avg": f"{avg_grad:.4f}",
                   "grad_max": f"{max_grad:.4f}",
                   **{k: f"{v:.6f}" if (
                       'loss' in k or 'mse' in k or 'std' in k or
                       'ratio' in k) else f"{v:.4f}"
                      for k, v in val.items()}}
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            spcc = val["val_fmri_spcc"]
            is_best = spcc > best_pcc
            logger.info(
                f"  VAL | v_cos={val['val_v_cos']:.4f} | "
                f"l_mse={val['val_latent_mse']:.4f} "
                f"l_pcc={val['val_latent_pcc']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} "
                f"f_spcc={spcc:.4f} | "
                f"z_std={val['val_zgen_std']:.4f}"
                f"{'  ★' if is_best else ''}")

            # Per-ROI PCC
            roi_str = " | ".join(
                f"{n}={val.get(f'roi_{n}_spcc', 0):.3f}"
                for n in roi_names)
            logger.info(f"  ROI | {roi_str}")

            # Layer mixing weights
            mix_w = eval_model.get_layer_mixing_weights()
            mix_row = {"epoch": epoch}
            w = mix_w['flow']
            for b in range(w.shape[0]):
                for li, layer_id in enumerate(dino_layer_names):
                    mix_row[f"flow_block{b}_layer{layer_id}"] = \
                        f"{w[b, li]:.4f}"
            with open(mixing_log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=mixing_fields).writerow(
                    mix_row)

            # Print mixing weights
            w = mix_w['flow']
            mix_str = " | ".join(
                f"B{b}:[" + ",".join(
                    f"{w[b, i]:.2f}" for i in range(w.shape[1])
                ) + "]" for b in range(w.shape[0]))
            logger.info(f"  FLOW_MIX | {mix_str}")

            if is_best:
                best_pcc = spcc
                patience_counter = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pcc": best_pcc, "config": cfg,
                    "layer_mixing_flow": mix_w['flow'].numpy()}
                torch.save(save_dict,
                    os.path.join(output_dir, "best_model.pt"))
                logger.info(f"  ★ Saved best (PCC={best_pcc:.4f})")
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        if epoch % train_cfg.get("save_every", 50) == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_pcc": best_pcc, "config": cfg}
            if use_ema:
                save_dict["ema_state_dict"] = ema_model.state_dict()
            torch.save(save_dict,
                os.path.join(output_dir, "latest.pt"))

    logger.info(f"Done! Best PCC: {best_pcc:.4f}")


if __name__ == "__main__":
    main()
