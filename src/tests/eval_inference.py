"""
Inference Evaluation Script — Stage 2 Flow Matching.

Tests multiple ODE solvers × timesteps × inference configurations.

Available solvers (torchdiffeq):
  Fixed-step:    euler, midpoint, rk4, heun3
  Adaptive-step: dopri5, dopri8, bosh3, adaptive_heun

Usage:
    python -m src.eval_inference \\
        --checkpoint results/subj01/stage2_guided_flow/best_model.pt \\
        --config src/configs/stage2_guided_flow_subj01.yaml

    # Sweep specific solvers and timesteps:
    python -m src.eval_inference \\
        --checkpoint ... \\
        --config ... \\
        --solvers euler midpoint rk4 dopri5 \\
        --ode_steps 50 100 150 200
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.model.residual_flow_sit import ResidualFlowSiT, ResidualFlowSiTConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig


# ─── Dataset ──────────────────────────────────────────────────────────────────


class FmriMultiLayerDataset(Dataset):
    def __init__(self, fmri_path, dino_path, max_samples=0):
        raw_fmri = np.load(fmri_path)
        self.dino_mmap = np.load(dino_path, mmap_mode='r')

        if raw_fmri.ndim == 3:
            fmri = raw_fmri.mean(axis=1).astype(np.float32)
        else:
            fmri = raw_fmri.astype(np.float32)
        del raw_fmri

        self.fmri = fmri[:max_samples] if max_samples > 0 else fmri
        self.n_samples = self.fmri.shape[0]
        print(f"  Evaluation set: {self.n_samples} samples")

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


class ODEWrapper(torch.nn.Module):
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


# Solvers que NO need fixed timesteps (adaptive)
ADAPTIVE_SOLVERS = {"dopri5", "dopri8", "bosh3", "adaptive_heun", "fehlberg2"}
# Fixed-step solvers
FIXED_STEP_SOLVERS = {"euler", "midpoint", "rk4", "heun3", "explicit_adams"}


# ─── Inference ────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_inference(model, vae, val_loader, device,
                  ode_steps=50, cfg_scale=1.0, num_trials=1,
                  prior_sigma=1.0, solver="euler"):
    from torchdiffeq import odeint

    is_adaptive = solver in ADAPTIVE_SOLVERS

    model.eval()
    all_pred, all_true = [], []
    all_z_gen, all_z_true = [], []
    all_traj_length, all_traj_straight = [], []

    for fmri, dino in val_loader:
        fmri, dino = fmri.to(device), dino.to(device)
        z1, _, _ = vae.encode(fmri, sample_posterior=False)

        ode_fn = ODEWrapper(model, dino, cfg_scale)

        z_gen = torch.zeros_like(z1)
        traj_len_total = 0.0
        straight_dist_total = 0.0

        for _ in range(num_trials):
            x0 = prior_sigma * torch.randn_like(z1)

            if is_adaptive:
                # Adaptive solvers only need start & end time; they choose steps
                t_span = torch.tensor([0.0, 1.0], device=device)
                traj = odeint(ode_fn, x0, t_span, method=solver,
                              rtol=1e-3, atol=1e-5)
            else:
                t_span = torch.linspace(0, 1, ode_steps, device=device)
                traj = odeint(ode_fn, x0, t_span, method=solver)

            step_diffs = traj[1:] - traj[:-1]
            traj_len = torch.norm(step_diffs, dim=-1).sum(dim=0).mean().item()
            straight_dist = torch.norm(traj[-1] - x0, dim=-1).mean().item()

            traj_len_total += traj_len
            straight_dist_total += straight_dist
            z_gen = z_gen + traj[-1]

        z_gen = z_gen / num_trials
        all_traj_length.append(traj_len_total / num_trials)
        all_traj_straight.append(straight_dist_total / num_trials)

        all_z_gen.append(z_gen)
        all_z_true.append(z1)
        all_pred.append(vae.decode(z_gen))
        all_true.append(fmri)

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)
    z_gens = torch.cat(all_z_gen)
    z_trues = torch.cat(all_z_true)

    avg_traj_len = sum(all_traj_length) / len(all_traj_length)
    avg_straight = sum(all_traj_straight) / len(all_traj_straight)

    return {
        "latent_mse": F.mse_loss(z_gens, z_trues).item(),
        "latent_pcc": pearson_corr_samplewise(z_gens, z_trues),
        "fmri_mse": F.mse_loss(preds, trues).item(),
        "fmri_vpcc": pearson_corr_voxelwise(preds, trues),
        "fmri_spcc": pearson_corr_samplewise(preds, trues),
        "z_gen_std": z_gens.std().item(),
        "traj_ratio": avg_traj_len / (avg_straight + 1e-8),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser("Inference Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--solvers", nargs="+",
                        default=["euler", "midpoint", "rk4", "heun3",
                                 "dopri5", "dopri8"],
                        help="List of ODE solvers to test")
    parser.add_argument("--ode_steps", nargs="+", type=int,
                        default=[50, 100, 150, 200],
                        help="Timesteps to test (fixed-step solvers only)")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Large batch = more GPU utilization (default 512)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for extra speed")
    parser.add_argument("--dtype", default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Model dtype: bf16 recommended for Ampere+ GPUs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    infer_dtype = dtype_map[args.dtype]
    print(f"Device: {device} | dtype: {args.dtype} | batch: {args.batch_size}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # ── Data ──
    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    root = data_cfg["root"]
    dino_suffix = data_cfg.get("dino_suffix", "dinov2_vitb14_multilayer")

    val_ds = FmriMultiLayerDataset(
        os.path.join(root, subject, f"nsd_test_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject, f"nsd_{dino_suffix}_test_sub{sub_num}.npy"),
        max_samples=args.max_samples)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    # ── VAE ──
    vae_ckpt = data_cfg["vae_checkpoint"]
    config_path = os.path.join(os.path.dirname(vae_ckpt), "config.yaml")
    if not os.path.exists(config_path):
        for guess in [f"src/configs/fmri_mlp_vae_768_{subject}.yaml",
                      "src/configs/fmri_mlp_vae_768_subj01.yaml"]:
            if os.path.exists(guess):
                config_path = guess
                break
    with open(config_path) as f:
        vae_cfg = yaml.safe_load(f)
    vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
    ckpt_vae = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt_vae["model_state_dict"])
    for p in vae.parameters():
        p.requires_grad = False
    print(f"VAE loaded from {vae_ckpt}")

    # ── Model ──
    model = ResidualFlowSiT(ResidualFlowSiTConfig(**model_cfg)).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print(f"Loaded EMA weights from {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model weights from {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    for p in model.parameters():
        p.requires_grad = False

    # ── Inference Config Grid (fixed per test) ──
    # Best configs from Phase 1 evaluation
    inf_configs = [
        # (cfg_scale, prior_sigma, num_trials, label)
        (1.0,  1.0,  1,  "σ=1.0 cfg=1.0 T=1"),   # Baseline
        (1.0,  0.3,  1,  "σ=0.3 cfg=1.0 T=1"),   # Best latent MSE
        (1.0,  1.0,  5,  "σ=1.0 cfg=1.0 T=5"),   # Best fMRI MSE
        (1.5,  0.5,  5,  "σ=0.5 cfg=1.5 T=5"),   # Best sPCC
    ]

    # ── Output ──
    COL = "{:<28} {:<10} {:<10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>8}"
    HDR = COL.format("Solver", "Steps", "Config",
                     "Lat-MSE", "Lat-PCC", "F-MSE", "F-vPCC", "F-sPCC",
                     "z-std", "Traj-R")
    SEP = "=" * len(HDR)
    print(f"\n{SEP}\n{HDR}\n{SEP}")

    out_dir = os.path.dirname(args.checkpoint)
    out_csv = os.path.join(out_dir, "eval_solvers.csv")
    fields = ["solver", "steps", "inf_config",
              "latent_mse", "latent_pcc", "fmri_mse",
              "fmri_vpcc", "fmri_spcc", "z_gen_std", "traj_ratio", "elapsed_s"]

    rows = []
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for solver in args.solvers:
            is_adaptive = solver in ADAPTIVE_SOLVERS
            steps_to_run = [None] if is_adaptive else args.ode_steps

            for steps in steps_to_run:
                steps_label = "adaptive" if is_adaptive else str(steps)

                for cfg_scale, prior_sigma, num_trials, inf_label in inf_configs:
                    t0 = time.time()
                    metrics = run_inference(
                        model, vae, val_loader, device,
                        ode_steps=steps if not is_adaptive else 2,
                        cfg_scale=cfg_scale,
                        num_trials=num_trials,
                        prior_sigma=prior_sigma,
                        solver=solver,
                    )
                    elapsed = time.time() - t0

                    print(COL.format(
                        solver, steps_label, inf_label,
                        f"{metrics['latent_mse']:.4f}",
                        f"{metrics['latent_pcc']:.4f}",
                        f"{metrics['fmri_mse']:.4f}",
                        f"{metrics['fmri_vpcc']:.4f}",
                        f"{metrics['fmri_spcc']:.4f}",
                        f"{metrics['z_gen_std']:.4f}",
                        f"{metrics['traj_ratio']:.4f}",
                    ))

                    row = {"solver": solver, "steps": steps_label,
                           "inf_config": inf_label,
                           "elapsed_s": f"{elapsed:.1f}",
                           **{k: f"{v:.6f}" for k, v in metrics.items()}}
                    writer.writerow(row)
                    rows.append(row)

                print(f"  --- [{solver} | steps={steps_label}] done ---")

    print(f"\n{SEP}")
    print(f"\nResults saved to: {out_csv}")

    # Print best row for each metric
    for metric_key, label in [("fmri_mse", "Best F-MSE (lowest)"),
                               ("fmri_spcc", "Best F-sPCC (highest)"),
                               ("latent_pcc", "Best Lat-PCC (highest)")]:
        fwd = metric_key in ("fmri_mse",)
        best = min(rows, key=lambda r: float(r[metric_key])) if fwd \
               else max(rows, key=lambda r: float(r[metric_key]))
        print(f"\n⭐ {label}: solver={best['solver']} steps={best['steps']}"
              f" cfg={best['inf_config']} → {best[metric_key]}")


if __name__ == "__main__":
    main()
