import argparse
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.brain_cond_flow import BrainCondFlow, BrainCondFlowConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig
from src.model.fmri_vit_vae import FmriViTVAE, create_fmri_vit_vae
from src.train_stage2_cond_ot_flow import FmriMultiLayerDataset, CondFlowODEWrapper, stochastic_euler_sample, pearson_corr_samplewise, pearson_corr_voxelwise
from src.utils.roi_utils import ROIDecomposer

@torch.no_grad()
def evaluate_and_plot(model, vae, val_loader, device, output_dir, ode_steps=50, cfg_scale=1.0, num_trials=1, decomposer=None, sampler="ode"):
    if sampler == "ode":
        from torchdiffeq import odeint

    model.eval()
    all_pred, all_true = [], []
    all_z_gen, all_z_true = [], []

    print("Generating predictions...")
    for fmri, dino in tqdm(val_loader):
        fmri, dino = fmri.to(device), dino.to(device)
        z1, _, _ = vae.encode(fmri, sample_posterior=False)

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

        all_z_gen.append(z_gen.cpu().numpy())
        all_z_true.append(z1.cpu().numpy())
        all_pred.append(fmri_pred.cpu().numpy())
        all_true.append(fmri.cpu().numpy())

    preds = np.concatenate(all_pred, axis=0) # (N, V)
    trues = np.concatenate(all_true, axis=0) # (N, V)
    
    # --- Metrics Computation ---
    print("Computing metrics...")
    pred_t = torch.tensor(preds)
    true_t = torch.tensor(trues)
    
    fmri_mse = F.mse_loss(pred_t, true_t).item()
    fmri_pcc_voxelwise = pearson_corr_voxelwise(pred_t, true_t)
    fmri_spcc = pearson_corr_samplewise(pred_t, true_t)
    
    spcc_per_sample = [] # list of spcc for each sample to plot hist
    for i in range(preds.shape[0]):
        sp = np.corrcoef(preds[i], trues[i])[0, 1]
        spcc_per_sample.append(sp)
    spcc_per_sample = np.array(spcc_per_sample)
    
    roi_metrics = {}
    if decomposer is not None:
        for roi in decomposer.rois:
            if roi.n_voxels > 10:
                p = pred_t[:, roi.indices]
                t = true_t[:, roi.indices]
                roi_metrics[f"roi_{roi.name}_spcc"] = pearson_corr_samplewise(p, t)

    print(f"FMRI MSE: {fmri_mse:.4f}")
    print(f"FMRI Voxelwise PCC: {fmri_pcc_voxelwise:.4f}")
    print(f"FMRI Samplewise PCC: {fmri_spcc:.4f}")
    
    # --- Plotting ---
    plots_dir = os.path.join(output_dir, "eval_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Generating plots in {plots_dir}...")

    # 1. Histogram of Sample-wise PCC
    plt.figure(figsize=(8, 5))
    plt.hist(spcc_per_sample, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Sample-wise PCC Distribution\nMean = {fmri_spcc:.4f}')
    plt.xlabel('PCC')
    plt.ylabel('Frequency (Samples)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'spcc_histogram.png'), dpi=150)
    plt.close()

    # 2. Global Distribution (Flattened Voxel Values)
    # To avoid memory issues, subsample voxels for the global histogram if huge
    subsample_idx = np.random.choice(preds.size, min(preds.size, 1000000), replace=False)
    plt.figure(figsize=(8, 5))
    plt.hist(trues.flatten()[subsample_idx], bins=100, alpha=0.5, label='True fMRI', color='blue', density=True)
    plt.hist(preds.flatten()[subsample_idx], bins=100, alpha=0.5, label='Predicted fMRI', color='orange', density=True)
    plt.title('Voxel Value Distribution (Subset)')
    plt.xlabel('fMRI Activation Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'voxel_value_distribution.png'), dpi=150)
    plt.close()

    # 3. Voxel-wise Mean Comparison (Amplitude)
    mean_true = np.mean(trues, axis=0) # (V,)
    mean_pred = np.mean(preds, axis=0) # (V,)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(mean_true, mean_pred, alpha=0.3, s=1)
    # Plot x=y line
    min_val = min(mean_true.min(), mean_pred.min())
    max_val = max(mean_true.max(), mean_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Voxel-wise Mean Comparison')
    plt.xlabel('True Voxel Mean')
    plt.ylabel('Predicted Voxel Mean')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'voxel_mean_scatter.png'), dpi=150)
    plt.close()

    # 4. Voxel-wise Variance Comparison (Spread)
    var_true = np.var(trues, axis=0)
    var_pred = np.var(preds, axis=0)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(var_true, var_pred, alpha=0.3, s=1, color='green')
    min_var = min(var_true.min(), var_pred.min())
    max_var = max(var_true.max(), var_pred.max())
    plt.plot([min_var, max_var], [min_var, max_var], 'r--')
    plt.title('Voxel-wise Variance Comparison')
    plt.xlabel('True Voxel Variance')
    plt.ylabel('Predicted Voxel Variance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'voxel_variance_scatter.png'), dpi=150)
    plt.close()

    # 5. Save Summary to txt
    with open(os.path.join(plots_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f"FMRI MSE: {fmri_mse:.4f}\n")
        f.write(f"FMRI Voxelwise PCC: {fmri_pcc_voxelwise:.4f}\n")
        f.write(f"FMRI Samplewise PCC: {fmri_spcc:.4f}\n\n")
        if roi_metrics:
            f.write("Per-ROI Samplewise PCC:\n")
            for k, v in roi_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")

    print(f"Evaluation finished! Results saved in {plots_dir}")


def main():
    parser = argparse.ArgumentParser("Evaluate Stage 2 Conditional OT Flow Matching")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or output_dir containing best_model.pt")
    parser.add_argument("--sampler", type=str, default=None, help="ode or stochastic_euler")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG Scale during sampling")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots")
    
    args = parser.parse_args()

    # Detect checkpoint or dir
    if os.path.isdir(args.checkpoint):
        ckpt_path = os.path.join(args.checkpoint, "best_model.pt")
        config_path = os.path.join(args.checkpoint, "config.yaml")
        output_dir = args.output_dir if args.output_dir else args.checkpoint
    else:
        ckpt_path = args.checkpoint
        config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
        output_dir = args.output_dir if args.output_dir else os.path.dirname(ckpt_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    sampler = args.sampler if args.sampler is not None else train_cfg.get("sampler", "ode")
    ode_steps = train_cfg.get("ode_steps", 50)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else train_cfg.get("cfg_scale", 1.0)
    num_trials = train_cfg.get("num_trials", 1)

    print(f"Using sampler: {sampler}, CFG Scale: {cfg_scale}, Output Dir: {output_dir}")

    # ── ROI Decomposer ──
    roi_dir = data_cfg.get("roi_dir", "Data/nsddata/ppdata/subj01/func1pt8mm/roi")
    decomposer = ROIDecomposer(roi_dir)

    # ── Data ──
    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    root = data_cfg["root"]
    dino_suffix = data_cfg.get("dino_suffix", "dinov2_vitl14_multilayer")

    print(f"Loading Test Set for {subject} ...")
    val_ds = FmriMultiLayerDataset(
        os.path.join(root, subject, f"nsd_test_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject, f"nsd_{dino_suffix}_test_sub{sub_num}.npy"),
        split="test", max_samples=0)

    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True)

    # ── Frozen VAE ──
    vae_ckpt = data_cfg["vae_checkpoint"]
    vae_config_path = os.path.join(os.path.dirname(vae_ckpt), "config.yaml")
    if not os.path.exists(vae_config_path):
        for alt in [
            f"src/configs/{subject}/stage1_vit_vae.yaml",
            f"src/configs/exp/fmri_mlp_vae_768_{subject}.yaml",
        ]:
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

    # ── Stage 2 Model ──
    model = BrainCondFlow(BrainCondFlowConfig(**model_cfg)).to(device)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
    # Note: if EMA was used during training, EMA weights might be in the checkpoint under 'ema_state_dict'. 
    # Usually we save `ema_state_dict` into "latest.pt", but for best_model.pt the primary is usually EMA.
    if "ema_state_dict" in ckpt_data:
        print("Using EMA state dict from checkpoint.")
        model.load_state_dict(ckpt_data["ema_state_dict"], strict=True)

    model.eval()
    print(f"Loaded Stage 2 model from {ckpt_path} (epoch {ckpt_data.get('epoch', '?')})")

    # Evaluate
    evaluate_and_plot(
        model, vae, val_loader, device, output_dir, 
        ode_steps=ode_steps, cfg_scale=cfg_scale, num_trials=num_trials, 
        decomposer=decomposer, sampler=sampler
    )

if __name__ == "__main__":
    main()
