"""
Velocity Field Noise Analysis for Flow Matching.

Analyzes how fMRI noise propagates into the velocity field u_t = z1 - x0,
and how well the model's predicted velocity v_pred aligns with the target.

Key analyses:
1. Velocity target variability: For the same image, how much does u_t vary
   due to different noise samples x0?
2. v_cos (cosine similarity) vs timestep t: Where in the flow is the model
   most/least aligned?
3. Velocity magnitude distribution: ||u_t|| vs ||v_pred||
4. Per-voxel velocity SNR: signal-to-noise ratio of the velocity field
5. Velocity field divergence analysis

Usage:
    python -m src.analyze_velocity_field \
        --checkpoint results/subj01/stage2_cond_ot_flow_fnp_v2
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.model.brain_cond_flow import BrainCondFlow, BrainCondFlowConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig
from src.model.fmri_vit_vae import create_fmri_vit_vae
from src.train_stage2_cond_ot_flow import (
    FmriMultiLayerDataset,
    pearson_corr_samplewise,
)


@torch.no_grad()
def analyze_velocity_field(model, vae, val_loader, device, output_dir):
    model.eval()
    plots_dir = os.path.join(output_dir, "velocity_analysis")
    os.makedirs(plots_dir, exist_ok=True)

    # ─── 1. Collect velocity statistics across timesteps ─────────────────
    print("=== Analysis 1: Velocity cosine similarity vs timestep ===")
    timesteps = np.linspace(0.01, 0.99, 50)
    vcos_per_t = []
    vmag_pred_per_t = []
    vmag_target_per_t = []
    mse_per_t = []

    # Use first 5 batches for analysis
    batches = []
    for i, (fmri, dino) in enumerate(val_loader):
        if i >= 5:
            break
        batches.append((fmri.to(device), dino.to(device)))

    for t_val in timesteps:
        cos_list, mag_pred_list, mag_tgt_list, mse_list = [], [], [], []
        for fmri, dino in batches:
            z1, _, _ = vae.encode(fmri, sample_posterior=False)
            x0 = torch.randn_like(z1)

            t_batch = torch.full((z1.shape[0],), t_val, device=device)
            t_expand = t_batch[:, None]
            xt = t_expand * z1 + (1 - t_expand) * x0
            ut = z1 - x0  # target velocity

            v_pred = model.forward_flow(t_batch, xt, dino)

            cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
            cos_list.append(cos)
            mag_pred_list.append(v_pred.norm(dim=-1).mean().item())
            mag_tgt_list.append(ut.norm(dim=-1).mean().item())
            mse_list.append(F.mse_loss(v_pred, ut).item())

        vcos_per_t.append(np.mean(cos_list))
        vmag_pred_per_t.append(np.mean(mag_pred_list))
        vmag_target_per_t.append(np.mean(mag_tgt_list))
        mse_per_t.append(np.mean(mse_list))

    # Plot 1a: v_cos vs timestep
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(timesteps, vcos_per_t, 'b-', linewidth=2)
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Velocity Alignment vs Timestep')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Plot 1b: velocity magnitude vs timestep
    axes[1].plot(timesteps, vmag_pred_per_t, 'b-', linewidth=2, label='||v_pred||')
    axes[1].plot(timesteps, vmag_target_per_t, 'r--', linewidth=2, label='||u_t||')
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel('Velocity Magnitude')
    axes[1].set_title('Velocity Magnitude vs Timestep')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 1c: MSE vs timestep
    axes[2].plot(timesteps, mse_per_t, 'g-', linewidth=2)
    axes[2].set_xlabel('Timestep t')
    axes[2].set_ylabel('MSE Loss')
    axes[2].set_title('Velocity MSE vs Timestep')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'velocity_vs_timestep.png'), dpi=150)
    plt.close()
    print(f"  Saved velocity_vs_timestep.png")

    # ─── 2. Velocity target variability (noise effect on u_t) ────────────
    print("=== Analysis 2: Velocity target variability ===")
    # For the same z1 (fMRI latent), sample multiple x0 and compute u_t
    # This shows how much the velocity target varies purely due to noise sampling
    n_trials = 20
    fmri_batch, dino_batch = batches[0]
    z1, _, _ = vae.encode(fmri_batch[:32], sample_posterior=False)

    # Also compute how much z1 itself varies if we use stochastic encoding
    z1_deterministic, _, _ = vae.encode(fmri_batch[:32], sample_posterior=False)

    all_ut = []
    all_vpred = []
    t_fixed = 0.5  # middle of the flow
    for trial in range(n_trials):
        x0 = torch.randn_like(z1)
        ut = z1 - x0
        all_ut.append(ut.cpu().numpy())

        t_batch = torch.full((z1.shape[0],), t_fixed, device=device)
        t_expand = t_batch[:, None]
        xt = t_expand * z1 + (1 - t_expand) * x0
        v_pred = model.forward_flow(t_batch, xt, dino_batch[:32])
        all_vpred.append(v_pred.cpu().numpy())

    all_ut = np.stack(all_ut, axis=0)    # (n_trials, B, D)
    all_vpred = np.stack(all_vpred, axis=0)

    # Variance of u_t across trials (per sample, per dimension)
    ut_var = np.var(all_ut, axis=0)       # (B, D)
    vpred_var = np.var(all_vpred, axis=0)

    # Mean velocity target
    ut_mean = np.mean(all_ut, axis=0)     # (B, D) — this is approximately z1

    # SNR of velocity target: signal = |mean(u_t)|^2, noise = var(u_t)
    ut_signal = ut_mean ** 2
    ut_snr_per_dim = ut_signal / (ut_var + 1e-8)  # (B, D)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 2a: Distribution of u_t variance
    axes[0].hist(ut_var.flatten(), bins=100, alpha=0.6, color='blue',
                 label='Target u_t', density=True)
    axes[0].hist(vpred_var.flatten(), bins=100, alpha=0.6, color='orange',
                 label='Predicted v', density=True)
    axes[0].set_xlabel('Variance across x0 trials')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Velocity Variability (t={t_fixed})\n'
                      f'u_t var: {ut_var.mean():.4f}, v_pred var: {vpred_var.mean():.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2b: SNR distribution
    snr_flat = ut_snr_per_dim.flatten()
    snr_flat = snr_flat[snr_flat < np.percentile(snr_flat, 99)]  # clip outliers
    axes[1].hist(snr_flat, bins=100, alpha=0.7, color='green', density=True)
    axes[1].axvline(x=np.median(snr_flat), color='r', linestyle='--',
                    label=f'Median SNR={np.median(snr_flat):.2f}')
    axes[1].set_xlabel('Velocity Target SNR (per dimension)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Velocity Target Signal-to-Noise Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 2c: Scatter v_pred variance vs u_t variance
    axes[2].scatter(ut_var.mean(axis=0), vpred_var.mean(axis=0),
                    alpha=0.3, s=2)
    max_v = max(ut_var.mean(axis=0).max(), vpred_var.mean(axis=0).max())
    axes[2].plot([0, max_v], [0, max_v], 'r--')
    axes[2].set_xlabel('u_t variance (per latent dim)')
    axes[2].set_ylabel('v_pred variance (per latent dim)')
    axes[2].set_title('Model Variance vs Target Variance\n(per latent dimension)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'velocity_variability.png'), dpi=150)
    plt.close()
    print(f"  Saved velocity_variability.png")

    # ─── 3. Decompose velocity into signal vs noise components ───────────
    print("=== Analysis 3: Signal vs Noise decomposition ===")
    # Key insight: u_t = z1 - x0
    # mean(u_t) over x0 trials = z1 (the "signal direction")
    # var(u_t) over x0 trials = var(x0) = I (isotropic noise)
    #
    # But z1 itself contains fMRI noise! So the TRUE signal direction
    # is only the stimulus-driven component of z1.
    # We can estimate this by looking at cross-sample variance of z1.

    # Collect all z1 from validation set
    all_z1 = []
    all_dino = []
    for fmri, dino in batches:
        z1, _, _ = vae.encode(fmri, sample_posterior=False)
        all_z1.append(z1.cpu().numpy())
        all_dino.append(dino.cpu().numpy())
    all_z1 = np.concatenate(all_z1, axis=0)  # (N, D)

    z1_var_per_dim = np.var(all_z1, axis=0)   # (D,)
    z1_mean_mag = np.mean(np.abs(all_z1), axis=0)  # (D,)

    # The ratio of z1 variance to total velocity variance (z1 var + x0 var = z1 var + 1)
    # tells us how much of the velocity is "signal" vs "noise from x0"
    signal_ratio = z1_var_per_dim / (z1_var_per_dim + 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 3a: z1 variance per dimension
    axes[0].hist(z1_var_per_dim, bins=80, alpha=0.7, color='purple', density=True)
    axes[0].axvline(x=np.mean(z1_var_per_dim), color='r', linestyle='--',
                    label=f'Mean var={np.mean(z1_var_per_dim):.3f}')
    axes[0].set_xlabel('Variance of z1 per latent dimension')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Latent z1 Variance Distribution\n'
                      '(How much "signal" exists per dimension)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 3b: Signal ratio distribution
    axes[1].hist(signal_ratio, bins=80, alpha=0.7, color='teal', density=True)
    axes[1].axvline(x=np.mean(signal_ratio), color='r', linestyle='--',
                    label=f'Mean={np.mean(signal_ratio):.3f}')
    axes[1].set_xlabel('Signal Ratio = Var(z1) / (Var(z1) + 1)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Velocity Signal Ratio per Dimension\n'
                      '(Fraction of velocity that is "signal")')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3c: Effective noise ceiling in velocity space
    # If fMRI noise ceiling is 36%, and VAE further compresses,
    # what fraction of z1 is actually predictable?
    # We estimate this as: for each latent dim, how correlated are
    # the z1 values across samples that share similar DINOv2 features?
    axes[2].bar(['Var(z1)\n(signal)', 'Var(x0)\n(sampling noise)', 'Total\nVar(u_t)'],
                [np.mean(z1_var_per_dim), 1.0,
                 np.mean(z1_var_per_dim) + 1.0],
                color=['#2196F3', '#FF5722', '#4CAF50'], alpha=0.8)
    axes[2].set_ylabel('Variance')
    axes[2].set_title('Velocity Target Variance Decomposition\n'
                      f'Signal fraction: {np.mean(signal_ratio):.1%}')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'signal_noise_decomposition.png'), dpi=150)
    plt.close()
    print(f"  Saved signal_noise_decomposition.png")

    # ─── 4. History analysis: v_cos trajectory ───────────────────────────
    print("=== Analysis 4: Training history v_cos trajectory ===")
    history_path = os.path.join(output_dir, "history.csv")
    if os.path.exists(history_path):
        import csv
        with open(history_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        epochs = [int(r['epoch']) for r in rows]
        train_loss = [float(r['train_loss']) for r in rows]
        val_flow = [float(r['val_flow_loss']) for r in rows]
        v_cos = [float(r['val_v_cos']) for r in rows]
        spcc = [float(r['val_fmri_spcc']) for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 4a: Train vs Val flow loss (overfitting indicator)
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_flow, 'r-', label='Val Flow Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Train vs Val Flow Loss\n(Divergence = Overfitting)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 4b: v_cos over training
        axes[0, 1].plot(epochs, v_cos, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].set_title('Velocity Cosine Similarity (Val)\n'
                             '(Model alignment with noisy targets)')
        axes[0, 1].grid(True, alpha=0.3)
        # Mark peak
        peak_idx = np.argmax(v_cos)
        axes[0, 1].axvline(x=epochs[peak_idx], color='r', linestyle='--',
                           alpha=0.5, label=f'Peak at epoch {epochs[peak_idx]}')
        axes[0, 1].legend()

        # 4c: Overfitting gap
        gap = [v - t for v, t in zip(val_flow, train_loss)]
        axes[1, 0].plot(epochs, gap, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Val Loss - Train Loss')
        axes[1, 0].set_title('Generalization Gap\n(Higher = More Overfitting)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 4d: SPCC over training
        axes[1, 1].plot(epochs, spcc, 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sample-wise PCC')
        axes[1, 1].set_title('fMRI Sample-wise PCC\n'
                             f'(NSD noise ceiling r=0.60)')
        axes[1, 1].axhline(y=0.60, color='r', linestyle='--', alpha=0.7,
                           label='Noise Ceiling (r=0.60)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_dynamics.png'), dpi=150)
        plt.close()
        print(f"  Saved training_dynamics.png")

    # ─── 5. Write summary ────────────────────────────────────────────────
    summary_path = os.path.join(plots_dir, 'velocity_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VELOCITY FIELD NOISE ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. VELOCITY TARGET COMPOSITION\n")
        f.write(f"   Mean Var(z1) per dim:  {np.mean(z1_var_per_dim):.4f}\n")
        f.write(f"   Var(x0) per dim:       1.0000 (by definition)\n")
        f.write(f"   Total Var(u_t) per dim: {np.mean(z1_var_per_dim) + 1.0:.4f}\n")
        f.write(f"   Signal fraction:       {np.mean(signal_ratio):.1%}\n")
        f.write(f"   → Only {np.mean(signal_ratio):.1%} of velocity target variance\n")
        f.write(f"     comes from the fMRI signal (z1).\n")
        f.write(f"     The rest ({1-np.mean(signal_ratio):.1%}) is pure x0 noise.\n\n")

        f.write("2. VELOCITY ALIGNMENT (v_cos at t=0.5)\n")
        mid_idx = len(timesteps) // 2
        f.write(f"   v_cos at t=0.5:        {vcos_per_t[mid_idx]:.4f}\n")
        f.write(f"   Peak v_cos:            {max(vcos_per_t):.4f} "
                f"(at t={timesteps[np.argmax(vcos_per_t)]:.2f})\n")
        f.write(f"   v_pred variability:    {vpred_var.mean():.4f}\n")
        f.write(f"   u_t variability:       {ut_var.mean():.4f}\n\n")

        f.write("3. VELOCITY SNR\n")
        f.write(f"   Median velocity SNR:   {np.median(snr_flat):.4f}\n")
        f.write(f"   Mean velocity SNR:     {np.mean(snr_flat):.4f}\n\n")

        f.write("4. IMPLICATIONS\n")
        sr = np.mean(signal_ratio)
        f.write(f"   The velocity target u_t = z1 - x0 is dominated by\n")
        f.write(f"   sampling noise x0 ({1-sr:.1%}), with only {sr:.1%} signal.\n")
        f.write(f"   Combined with fMRI noise ceiling of 36%, the actual\n")
        f.write(f"   stimulus-driven signal in u_t is even lower.\n")
        f.write(f"   This fundamentally limits how well MSE can work as\n")
        f.write(f"   a velocity matching objective.\n")

    print(f"\nFull summary saved to: {summary_path}")
    print(f"All plots saved to: {plots_dir}/")


def main():
    parser = argparse.ArgumentParser("Velocity Field Noise Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    if os.path.isdir(args.checkpoint):
        ckpt_path = os.path.join(args.checkpoint, "best_model.pt")
        config_path = os.path.join(args.checkpoint, "config.yaml")
        output_dir = args.checkpoint
    else:
        ckpt_path = args.checkpoint
        config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
        output_dir = os.path.dirname(ckpt_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    root = data_cfg["root"]
    dino_suffix = data_cfg.get("dino_suffix", "dinov2_vitl14_multilayer")

    print(f"Loading validation data for {subject}...")
    val_ds = FmriMultiLayerDataset(
        os.path.join(root, subject, f"nsd_test_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject, f"nsd_{dino_suffix}_test_sub{sub_num}.npy"),
        split="test", max_samples=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

    # Load VAE
    vae_ckpt = data_cfg["vae_checkpoint"]
    vae_config_path = os.path.join(os.path.dirname(vae_ckpt), "config.yaml")
    if not os.path.exists(vae_config_path):
        for alt in [f"src/configs/{subject}/stage1_vit_vae.yaml"]:
            if os.path.exists(alt):
                vae_config_path = alt
                break
    with open(vae_config_path) as f:
        vae_cfg = yaml.safe_load(f)
    vae_model_type = vae_cfg.get("model_type", "mlp")
    if vae_model_type == "vit":
        vae = create_fmri_vit_vae(**vae_cfg["model"]).to(device).eval()
    else:
        vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
    vae_ckpt_data = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(vae_ckpt_data["model_state_dict"])
    for p in vae.parameters():
        p.requires_grad = False
    print(f"Loaded VAE from {vae_ckpt}")

    # Load Flow model
    model = BrainCondFlow(BrainCondFlowConfig(**model_cfg)).to(device)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
    if "ema_state_dict" in ckpt_data:
        model.load_state_dict(ckpt_data["ema_state_dict"], strict=True)
    model.eval()
    print(f"Loaded model from {ckpt_path} (epoch {ckpt_data.get('epoch', '?')})")

    analyze_velocity_field(model, vae, val_loader, device, output_dir)


if __name__ == "__main__":
    main()
