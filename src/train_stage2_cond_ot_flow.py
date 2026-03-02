"""
Stage 2: Direct Conditional OT Flow Matching — DINOv2 → fMRI latent.

Standard conditional flow matching from N(0,I) to z_true, conditioned on
DINOv2 features. No regression branch, no residual decomposition.

Inspired by MOTFM (Yazdani et al., MICCAI 2025):
  - Conditional OT straight-line path: x_t = (1-t)*x_0 + t*x_1
  - Velocity target: u_t = x_1 - x_0
  - Loss = MSE(v_pred, u_t)

Usage:
    python -m src.train_stage2_cond_ot_flow --config src/configs/subj01/stage2_cond_ot_flow_vit_vae.yaml
    python -m src.train_stage2_cond_ot_flow --config src/configs/subj01/stage2_cond_ot_flow_vit_vae.yaml --debug
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
from torch.utils.data import DataLoader, Dataset

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from src.model.brain_cond_flow import BrainCondFlow, BrainCondFlowConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig
from src.model.fmri_vit_vae import FmriViTVAE, create_fmri_vit_vae
from src.model.fmri_roi_vae import FmriROIVAE, create_fmri_roi_vae
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


# ─── Velocity Loss Functions ─────────────────────────────────────────────────


def compute_velocity_loss(v_pred, ut, t=None, loss_type="mse",
                          huber_c=0.1,
                          timestep_weight_type="none",
                          direction_weight=1.0,
                          magnitude_weight=0.1):
    """
    Compute velocity matching loss with multiple strategies.

    Args:
        v_pred: (B, D) predicted velocity
        ut: (B, D) target velocity
        t: (B,) timestep values, needed for timestep weighting
        loss_type: "mse" | "pseudo_huber" | "decoupled"
        huber_c: threshold for Pseudo-Huber transition from L2 to L1
        timestep_weight_type: "none" | "linear" | "cosine" | "snr"
        direction_weight: weight for cosine direction loss (decoupled)
        magnitude_weight: weight for magnitude loss (decoupled)

    Returns:
        loss: scalar loss value
        loss_info: dict with component values for logging
    """
    info = {}

    if loss_type == "pseudo_huber":
        # Pseudo-Huber: sqrt((v-u)^2 + c^2) - c
        # Behaves like MSE near 0, like L1 for large errors
        diff_sq = (v_pred - ut).pow(2)
        per_element = (diff_sq + huber_c ** 2).sqrt() - huber_c
        raw_loss = per_element.mean(dim=-1)  # (B,)
        info["huber_raw"] = raw_loss.mean().item()

    elif loss_type == "decoupled":
        # Separate direction and magnitude learning
        # Direction: cosine similarity loss
        cos_sim = F.cosine_similarity(v_pred, ut, dim=-1)  # (B,)
        dir_loss = 1.0 - cos_sim  # (B,)

        # Magnitude: log-ratio loss (scale-invariant)
        pred_mag = v_pred.norm(dim=-1).clamp(min=1e-6)  # (B,)
        tgt_mag = ut.norm(dim=-1).clamp(min=1e-6)       # (B,)
        mag_loss = (pred_mag.log() - tgt_mag.log()).pow(2)  # (B,)

        raw_loss = direction_weight * dir_loss + magnitude_weight * mag_loss
        info["dir_loss"] = dir_loss.mean().item()
        info["mag_loss"] = mag_loss.mean().item()
        info["v_cos"] = cos_sim.mean().item()

    else:  # "mse" (default)
        raw_loss = (v_pred - ut).pow(2).mean(dim=-1)  # (B,)

    # ─── Timestep weighting ───
    if t is not None and timestep_weight_type != "none":
        if timestep_weight_type == "linear":
            # w(t) = 1 - t:  full weight at t=0, zero at t=1
            w = 1.0 - t
        elif timestep_weight_type == "cosine":
            # w(t) = cos(π*t/2)²: smooth decay
            w = torch.cos(t * 3.14159 / 2).pow(2)
        elif timestep_weight_type == "snr":
            # w(t) = 1/(1 + t/(1-t+ε)): SNR-inspired weighting
            w = 1.0 / (1.0 + t / (1.0 - t + 1e-4))
        else:
            w = torch.ones_like(t)

        w = w / (w.mean() + 1e-8)  # normalize so mean weight ≈ 1
        loss = (w * raw_loss).mean()
        info["w_mean"] = w.mean().item()
        info["w_std"] = w.std().item()
    else:
        loss = raw_loss.mean()

    return loss, info


# ─── ODE Wrapper ─────────────────────────────────────────────────────────────


class CondFlowODEWrapper(torch.nn.Module):
    """ODE wrapper: model predicts velocity for direct flow.
    z_gen = integration from x0 (noise) to z_true."""

    def __init__(self, model, context, cfg_scale=1.0):
        super().__init__()
        self.model = model
        self.context = context
        self.cfg_scale = cfg_scale

    def forward(self, t, z):
        B = z.shape[0]
        t_batch = t.expand(B)
        if self.cfg_scale == 1.0:
            return self.model.forward_flow(t_batch, z, self.context)
        else:
            return self.model.forward_flow_with_cfg(
                t_batch, z, self.context, self.cfg_scale)


@torch.no_grad()
def stochastic_euler_sample(model, x0, context, T, cfg_scale=1.0):
    """FlowNP-style stochastic Euler sampling.
    Uses dampening schedule alpha = 1 + t*(1-t) and noise injection
    sigma = 0.2 * sqrt(t*(1-t)) for better sample diversity."""
    z = x0
    B = z.shape[0]
    for i in range(T):
        tt = i / T
        t_batch = torch.full((B,), tt, device=z.device)
        if cfg_scale == 1.0:
            v_pred = model.forward_flow(t_batch, z, context)
        else:
            v_pred = model.forward_flow_with_cfg(t_batch, z, context, cfg_scale)
        alpha = 1 + tt * (1 - tt)
        sigma = 0.2 * (tt * (1 - tt)) ** 0.5
        z = z + (alpha * v_pred + sigma * torch.randn_like(z)) / T
    return z


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, vae, val_loader, fm, device, ode_steps=50,
             cfg_scale=1.0, num_trials=1, decomposer=None,
             sampler="ode"):
    if sampler == "ode":
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

        # Flow loss (x0=noise → z_true)
        x0 = torch.randn_like(z1)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, z1)
        v_pred = model.forward_flow(t, xt, dino)

        flow_loss = F.mse_loss(v_pred, ut)
        total_flow_loss += flow_loss.item()
        n_batches += 1

        cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
        all_v_cos.append(cos)

        # Generation: noise -> z_gen
        z_gen = torch.zeros_like(z1)
        for _ in range(num_trials):
            x0_trial = torch.randn_like(z1)
            if sampler == "stochastic_euler":
                z_trial = stochastic_euler_sample(
                    model, x0_trial, dino, T=ode_steps, cfg_scale=cfg_scale)
            else:
                ode_fn = CondFlowODEWrapper(model, dino, cfg_scale)
                t_span = torch.linspace(0, 1, ode_steps, device=device)
                traj = odeint(ode_fn, x0_trial, t_span, method="midpoint")
                z_trial = traj[-1]
            z_gen += z_trial
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
        # Generation quality
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
                t = trues[:, roi.indices]
                metrics[f"roi_{roi.name}_spcc"] = pearson_corr_samplewise(
                    p, t)
            else:
                metrics[f"roi_{roi.name}_spcc"] = 0.0

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        "Stage 2: Direct Conditional OT Flow Matching")
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
    cfg_drop_prob = train_cfg.get("cfg_drop_prob", 0.1)
    cfg_scale = train_cfg.get("cfg_scale", 1.0)
    ode_steps = train_cfg.get("ode_steps", 50)
    num_trials = train_cfg.get("num_trials", 1)
    eval_interval = 1 if args.debug else train_cfg.get("eval_interval", 5)
    sampler = train_cfg.get("sampler", "ode")

    # Flow options
    use_ot = train_cfg.get("use_ot", True)
    context_mask_ratio = train_cfg.get("context_mask_ratio", 0.3)

    timestep_sampling = train_cfg.get("timestep_sampling", "logit_normal")
    logit_normal_mu = train_cfg.get("logit_normal_mu", 0.0)
    logit_normal_sigma = train_cfg.get("logit_normal_sigma", 1.0)

    # Loss options
    velocity_loss_type = train_cfg.get("velocity_loss_type", "mse")
    huber_c = train_cfg.get("huber_c", 0.1)
    timestep_weight_type = train_cfg.get("timestep_weight_type", "none")
    direction_weight = train_cfg.get("direction_weight", 1.0)
    magnitude_weight = train_cfg.get("magnitude_weight", 0.1)

    output_dir = cfg.get("output_dir", "results/stage2_cond_ot_flow")
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
    logger = logging.getLogger('stage2_cond_ot_flow')
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
    if vae_model_type == "roi":
        vae = create_fmri_roi_vae(**vae_cfg["model"]).to(device).eval()
        logger.info("VAE type: ROI")
    elif vae_model_type == "vit":
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

    # ── Model ──
    model = BrainCondFlow(BrainCondFlowConfig(**model_cfg)).to(device)
    ema_model = copy.deepcopy(model) if use_ema else None
    pc = model.param_count()
    logger.info(
        f"BrainCondFlow: ctx={pc['ctx_M']:.1f}M "
        f"flow={pc['flow_M']:.1f}M total={pc['total_M']:.1f}M | "
        f"backbone={model_cfg.get('backbone', 'dit')} | EMA={'ON' if use_ema else 'OFF'}")

    # ── Flow Matcher (Minibatch Optimal Transport) ──
    sigma = train_cfg.get("sigma", 0.0)
    if use_ot:
        try:
            import ot
            fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
            logger.info(f"Using Minibatch ExactOptimalTransport Flow Matcher (sigma={sigma})")
        except ImportError:
            logger.warning("POT library not found (`pip install POT`). Reverting to standard Flow Matching.")
            fm = ConditionalFlowMatcher(sigma=sigma)
            use_ot = False
    else:
        logger.info(f"Using standard Conditional Flow Matcher (sigma={sigma})")
        fm = ConditionalFlowMatcher(sigma=sigma)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.05))

    # ── History ──
    history_path = os.path.join(output_dir, "history.csv")
    roi_fields = [f"roi_{n}_spcc" for n in roi_names]
    fields = [
        "epoch", "train_loss", "lr", "grad_avg", "grad_max",
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
    dino_layer_names = cfg.get("dino_layers", [6, 12, 18, 24])
    mixing_fields = ["epoch"]
    for l in dino_layer_names:
        mixing_fields.append(f"shared_layer{l}")
    with open(mixing_log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=mixing_fields).writeheader()

    # ── Training ──
    ts_info = f"timestep_sampling={timestep_sampling}"
    if timestep_sampling == "logit_normal":
        ts_info += f" (mu={logit_normal_mu}, sigma={logit_normal_sigma})"
    logger.info(
        f"Training {num_epochs} epochs, eval every {eval_interval} | {ts_info}")
    logger.info(
        f"DIRECT COND FLOW | OT={use_ot} ContextMask={context_mask_ratio} Sampler={sampler}")
    logger.info(
        f"LOSS: type={velocity_loss_type} | t_weight={timestep_weight_type}"
        + (f" | huber_c={huber_c}" if velocity_loss_type == "pseudo_huber" else "")
        + (f" | dir_w={direction_weight} mag_w={magnitude_weight}" if velocity_loss_type == "decoupled" else "")
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(
            optimizer, epoch - 1, num_epochs, warmup_epochs, lr)

        ep_total, n_steps = 0, 0
        grads_all, grads_max_all = [], []
        t0 = time.time()

        for batch_idx, (fmri, dino) in enumerate(train_loader):
            fmri, dino = fmri.to(device), dino.to(device)
            B = fmri.shape[0]

            with torch.no_grad():
                z1, _, _ = vae.encode(fmri, sample_posterior=False)

            # ─── Flow Matching: x0 = noise, target = z_true ─────
            x0 = torch.randn_like(z1)

            # CFG dropout on context
            context = dino
            if cfg_drop_prob > 0:
                drop = torch.rand(B, device=device) < cfg_drop_prob
                if drop.any():
                    context = context.clone()
                    context[drop] = 0.0

            # Timestep sampling
            if timestep_sampling == "logit_normal":
                u = torch.randn(B, device=device)
                t_sample = torch.sigmoid(
                    logit_normal_mu + logit_normal_sigma * u)
                t_expand = t_sample[:, None]
                xt = t_expand * z1 + (1 - t_expand) * x0
                ut = z1 - x0
                t = t_sample
            else:
                t, xt, ut = fm.sample_location_and_conditional_flow(
                    x0, z1)

            # Forward: predict velocity
            v_pred = model.forward_flow(t, xt, context, mask_ratio=context_mask_ratio)

            # ─── Loss: Velocity Matching ─────────────────────────
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
                    v_cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
                logger.info(
                    f"  [Ep{epoch} B0] loss={loss.item():.4f} "
                    f"v_cos={v_cos:.4f}")

        avg_total = ep_total / max(n_steps, 1)
        avg_grad = sum(grads_all) / len(grads_all)
        max_grad = max(grads_max_all)
        ep_time = time.time() - t0

        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) [COND FLOW] | "
            f"loss={avg_total:.5f} | "
            f"lr={current_lr:.2e} grad={avg_grad:.4f}")

        # ── Eval ──
        if epoch % eval_interval == 0 or epoch == 1:
            eval_model = ema_model if use_ema else model
            val = validate(eval_model, vae, val_loader, fm, device,
                           ode_steps, cfg_scale, num_trials,
                           decomposer=decomposer, sampler=sampler)

            row = {"epoch": epoch,
                   "train_loss": f"{avg_total:.6f}",
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
                f"GEN: l_mse={val['val_latent_mse']:.4f} l_pcc={val['val_latent_pcc']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} f_spcc={spcc:.4f} | "
                f"z_std={val['val_zgen_std']:.4f}{'  ★' if is_best else ''}")

            # Per-ROI PCC
            roi_str = " | ".join(
                f"{n}={val.get(f'roi_{n}_spcc', 0):.3f}"
                for n in roi_names)
            logger.info(f"  ROI | {roi_str}")

            # Layer mixing weights
            mix_w = eval_model.get_layer_mixing_weights()
            mix_row = {"epoch": epoch}
            w = mix_w['shared']
            for li, l in enumerate(dino_layer_names):
                mix_row[f"shared_layer{l}"] = f"{w[0, li]:.4f}"
            with open(mixing_log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=mixing_fields).writerow(mix_row)

            # Print mixing weights
            mix_str = " | ".join(f"[{w[0, i]:.2f}]" for i in range(w.shape[1]))
            logger.info(f"  SHARED_MIX | {mix_str}")

            if is_best:
                best_pcc = spcc
                patience_counter = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pcc": best_pcc, "config": cfg,
                    "layer_mixing_shared": mix_w['shared'].numpy()}
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
