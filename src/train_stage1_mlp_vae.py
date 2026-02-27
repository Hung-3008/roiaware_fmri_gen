"""
Training script for fMRI MLP VAE (Stage 1).

Architecture:
    fMRI [B, 15724] → MLP Encoder → z ~ N(μ, σ²) [B, 1024]
    → MLP Decoder → fMRI_recon [B, 15724]

Usage:
    python -m src.train_stage1_mlp_vae --config src/configs/fmri_mlp_vae.yaml
    python -m src.train_stage1_mlp_vae --config src/configs/fmri_mlp_vae.yaml --debug
"""

import argparse
import csv
import logging
import time
import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from src.model.fmri_mlp_vae import FmriMLPVAE, create_fmri_mlp_vae
from src.model.fmri_vit_vae import FmriViTVAE, create_fmri_vit_vae
from src.utils.metrics import pearson_correlation
from src.utils.training import (
    EarlyStopping, CosineAnnealingWithWarmup,
    save_checkpoint, load_checkpoint, setup_logger, set_seed,
)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FmriFlatDataset(Dataset):
    """
    Dataset for NSD fMRI data.

    Handles fMRI arrays with shape (N, 3, V) where 3 = repetitions.

    Args:
        fmri_path: path to .npy file, shape (N, 3, V) or (N, V)
        split: 'train' or 'test' (for logging)
        rep_mode: how to handle 3 repetitions:
            'average'      → mean of 3 reps → N samples of shape (V,)
            'single_trial'  → each rep is a sample → 3N samples of shape (V,)
            'first'         → use only the first rep → N samples of shape (V,)
    """

    def __init__(self, fmri_path: str, split: str = "train", rep_mode: str = "average"):
        raw = np.load(fmri_path, mmap_mode='r')
        self.rep_mode = rep_mode

        if raw.ndim == 3:  # (N, 3, V)
            n_samples, n_reps, n_voxels = raw.shape
            if rep_mode == "average":
                # Mean of 3 reps → (N, V)
                self.fmri = np.mean(raw, axis=1)  # forces load into RAM
                print(f"FmriFlatDataset [{split}]: (N={n_samples}, reps={n_reps}, V={n_voxels}) "
                      f"→ averaged to ({self.fmri.shape[0]}, {self.fmri.shape[1]})")
            elif rep_mode == "single_trial":
                # Each rep as separate sample → (3N, V)
                self.fmri = raw.reshape(-1, n_voxels)  # forces load into RAM
                print(f"FmriFlatDataset [{split}]: (N={n_samples}, reps={n_reps}, V={n_voxels}) "
                      f"→ single_trial ({self.fmri.shape[0]}, {self.fmri.shape[1]})")
            elif rep_mode == "first":
                self.fmri = raw[:, 0, :]  # first rep only
                print(f"FmriFlatDataset [{split}]: (N={n_samples}, reps={n_reps}, V={n_voxels}) "
                      f"→ first rep ({self.fmri.shape[0]}, {self.fmri.shape[1]})")
            else:
                raise ValueError(f"Unknown rep_mode: {rep_mode}")
        elif raw.ndim == 2:  # (N, V) — already flat
            self.fmri = raw
            print(f"FmriFlatDataset [{split}]: shape={self.fmri.shape} (already flat)")
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {raw.shape}")

    def __len__(self):
        return self.fmri.shape[0]

    def __getitem__(self, idx):
        fmri = torch.from_numpy(np.array(self.fmri[idx])).float()
        return {"fmri": fmri}


# ─── Beta Schedule ────────────────────────────────────────────────────────────

def get_beta(epoch: int, beta_max: float, beta_anneal_epochs: int) -> float:
    """Linear β-annealing: beta_min → beta_max over beta_anneal_epochs.

    Bug fix: uses (epoch+1) instead of epoch so epoch-0 gets a non-zero beta.
    With epoch=0 the old formula gave beta=0 → pure autoencoder → posterior collapse.
    """
    if beta_anneal_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * (epoch + 1) / beta_anneal_epochs)


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, beta, lambda_pcc=0.5) -> dict:
    """Validate and compute metrics (deterministic: z = mu)."""
    model.eval()

    total = defaultdict(float)
    n = 0

    for batch in val_loader:
        fmri = batch["fmri"].to(device)

        # Deterministic encoding (no sampling)
        z, mu, logvar = model.encode(fmri, sample_posterior=False)
        x_recon = model.decode(z)

        losses = model.compute_loss(fmri, x_recon, mu, logvar, beta, lambda_pcc)

        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                total[k] += v.item() * fmri.shape[0]
            else:
                total[k] += v * fmri.shape[0]
        n += fmri.shape[0]

    avg = {k: v / max(n, 1) for k, v in total.items()}

    # Overall PCC (voxel-wise)
    return avg


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, train_loader, optimizer, scheduler, scaler, device,
    beta, lambda_pcc, grad_clip, use_amp, log_interval, logger, epoch,
) -> dict:
    """Train one epoch."""
    model.train()

    total = defaultdict(float)
    # Bug fix: track n_samples (not n_batches) so train metrics are per-sample,
    # consistent with validate() which also aggregates per-sample.
    n_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        fmri = batch["fmri"].to(device)
        bs = fmri.shape[0]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
            x_recon, z, mu, logvar = model(fmri)
            losses = model.compute_loss(fmri, x_recon, mu, logvar, beta, lambda_pcc)
            loss = losses["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            if logger:
                logger.warning(f"  NaN/Inf loss at batch {batch_idx+1}, skipping")
            optimizer.zero_grad()
            n_samples += bs  # still count samples to avoid skewing the average
            continue

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                total[k] += v.item() * bs
            else:
                total[k] += v * bs
        n_samples += bs

        if logger and (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"  Ep {epoch} [{batch_idx+1}/{len(train_loader)}] | "
                f"loss={loss.item():.4f} mse={losses['mse'].item():.4f} "
                f"kl={losses['kl'].item():.2f} pcc={losses['pcc'].item():.4f} β={beta:.4f}"
            )

    return {k: v / max(n_samples, 1) for k, v in total.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train fMRI MLP VAE (Stage 1)")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--device", type=str, default="", help="Device")
    parser.add_argument("--resume", type=str, default="", help="Resume checkpoint")
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["training"]["epochs"] = 3
        cfg["training"]["batch_size"] = 4

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(cfg["training"].get("seed", 42))

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("mlp_vae", save_dir)

    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Data ──
    data_cfg = cfg["data"]
    subject = data_cfg.get("subject", "subj01")    # e.g. "subj01"
    sub_num = int(subject.replace("subj", "").lstrip("0"))  # e.g. 1
    data_root = Path(data_cfg["root"])

    # Repetition handling mode
    train_rep_mode = data_cfg.get("train_rep_mode", "single_trial")
    test_rep_mode = data_cfg.get("test_rep_mode", "average")

    # NSD naming convention: Data/nsd/subj01/nsd_{split}_fmri_zscore_sub{N}.npy
    train_fmri_path = str(data_root / subject / f"nsd_train_fmri_zscore_sub{sub_num}.npy")
    test_fmri_path = str(data_root / subject / f"nsd_test_fmri_zscore_sub{sub_num}.npy")

    logger.info(f"Train fMRI: {train_fmri_path} (rep_mode={train_rep_mode})")
    logger.info(f"Test fMRI:  {test_fmri_path} (rep_mode={test_rep_mode})")

    train_dataset = FmriFlatDataset(train_fmri_path, "train", rep_mode=train_rep_mode)
    test_dataset = FmriFlatDataset(test_fmri_path, "test", rep_mode=test_rep_mode)

    if args.debug:
        train_dataset = Subset(train_dataset, list(range(min(32, len(train_dataset)))))
        test_dataset = Subset(test_dataset, list(range(min(16, len(test_dataset)))))

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ──
    model_cfg = cfg["model"]
    model_type = cfg.get("model_type", "mlp")

    if model_type == "vit":
        model = create_fmri_vit_vae(**model_cfg).to(device)
        model_name = "FmriViTVAE"
    else:
        model = create_fmri_mlp_vae(**model_cfg).to(device)
        model_name = "FmriMLPVAE"

    params = model.param_count()
    logger.info(f"{model_name}: {params['total']:,} params ({params['total_mb']:.1f} MB)")

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["training"].get("use_amp", True))

    total_epochs = cfg["training"]["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * 5
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # ── Loss config ──
    loss_cfg = cfg.get("loss", {})
    beta_max = loss_cfg.get("beta_max", 0.01)
    beta_anneal_epochs = loss_cfg.get("beta_anneal_epochs", 30)
    lambda_pcc = loss_cfg.get("lambda_pcc", 0.5)

    # ── Resume ──
    start_epoch = 0
    best_val_pcc = -float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Bug fix: restore scheduler state so LR continues from where it left off.
        # Without this the scheduler resets to warmup, causing an LR spike on resume.
        if "scheduler_state_dict" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_pcc = ckpt.get("best_val_pcc", -float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    # ── CSV Logger ──
    csv_path = save_dir / "history.csv"
    csv_header = [
        "epoch", "train_loss", "train_mse", "train_kl", "train_pcc",
        "val_loss", "val_mse", "val_kl", "val_pcc", "beta",
    ]
    mode = "a" if args.resume and csv_path.exists() else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(csv_header)

    es_patience = cfg["training"].get("early_stopping_patience", 50)
    early_stopping = EarlyStopping(
        patience=es_patience,
        mode="max",
    ) if es_patience > 0 else None

    # ── Training Loop ──
    logger.info(f"Training for {total_epochs} epochs...")
    logger.info(f"  Train: {len(train_dataset)}, Val: {len(test_dataset)}")
    logger.info(f"  Steps/epoch: {steps_per_epoch}, β_max: {beta_max}")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        beta = get_beta(epoch, beta_max, beta_anneal_epochs)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            beta, lambda_pcc,
            cfg["training"].get("grad_clip", 1.0),
            cfg["training"].get("use_amp", True),
            cfg["training"].get("log_interval", 50),
            logger, epoch,
        )

        val_metrics = validate(model, val_loader, device, beta, lambda_pcc)

        ep_time = time.time() - t0
        val_pcc = val_metrics.get("pcc", 0)
        logger.info(
            f"Epoch {epoch}/{total_epochs} ({ep_time:.1f}s) | "
            f"Train: loss={train_metrics.get('loss', 0):.4f} mse={train_metrics.get('mse', 0):.4f} "
            f"kl={train_metrics.get('kl', 0):.2f} pcc={train_metrics.get('pcc', 0):.4f} | "
            f"Val: loss={val_metrics.get('loss', 0):.4f} mse={val_metrics.get('mse', 0):.4f} "
            f"kl={val_metrics.get('kl', 0):.2f} pcc={val_pcc:.4f} | β={beta:.4f}"
        )

        # CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_metrics.get("loss", 0), train_metrics.get("mse", 0),
                train_metrics.get("kl", 0), train_metrics.get("pcc", 0),
                val_metrics.get("loss", 0), val_metrics.get("mse", 0),
                val_metrics.get("kl", 0), val_pcc, beta,
            ])

        # Save best
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            save_checkpoint(
                model, optimizer, epoch, train_metrics,
                save_dir / "best.pt", scheduler=scheduler,
                extra={"best_val_pcc": best_val_pcc},
            )
            logger.info(f"  ★ New best PCC: {best_val_pcc:.4f}")

        if early_stopping is not None:
            early_stopping(val_pcc)
            if early_stopping.should_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Bug fix: guard against NameError when the training loop never ran
    # (e.g. start_epoch == total_epochs on resume).
    if 'epoch' in dir() or 'epoch' in locals():
        save_checkpoint(
            model, optimizer, epoch, train_metrics,
            save_dir / "final.pt", scheduler=scheduler,
            extra={"best_val_pcc": best_val_pcc},
        )
    else:
        logger.warning("No epochs were run; skipping final checkpoint.")

    logger.info(f"Training complete. Best val PCC: {best_val_pcc:.4f}")
    logger.info(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
