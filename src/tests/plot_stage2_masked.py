import pandas as pd
import matplotlib.pyplot as plt

history_file = "results/subj01/stage2_masked_vit_vae/history.csv"
df = pd.read_csv(history_file)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Stage 2 Masked Brain Modeling Diagnostic (ViT VAE)", fontsize=16)

# 1. Losses
ax = axes[0, 0]
ax.plot(df['epoch'], df['train_loss'], label='Train Masked Loss', color='blue')
ax.plot(df['epoch'], df['val_loss'], label='Val Full Loss', color='red', linestyle='--')
ax.set_title("Training vs Validation Loss")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# 2. Latent Space Accuracy (PCC)
ax = axes[0, 1]
ax.plot(df['epoch'], df['val_latent_pcc'], label='Val Latent PCC', color='green')
ax.axhline(0.33, color='gray', linestyle='--', alpha=0.5, label='Old Flow Plateau (0.33)')
ax.set_title("Latent Space PCC (Generative Accuracy)")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# 3. fMRI Space Accuracy (PCC)
ax = axes[1, 0]
ax.plot(df['epoch'], df['val_fmri_pcc'], label='Val Voxelwise PCC', color='purple')
ax.plot(df['epoch'], df['val_fmri_spcc'], label='Val Samplewise PCC', color='magenta')
ax.set_title("Target fMRI Space (Reconstruction Quality)")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# 4. Latent Variance
ax = axes[1, 1]
ax.plot(df['epoch'], df['val_zgen_std'], label='Gen Latent Std', color='brown')
ax.plot(df['epoch'], df['val_zgen_crossvar_ratio'], label='Cross Var Ratio', color='orange')
ax.set_title("Latent Variance Metrics")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = "results/subj01/stage2_masked_vit_vae/diagnostic_masked.png"
plt.savefig(out_path)
print(f"Saved diagnostic plot to {out_path}")
