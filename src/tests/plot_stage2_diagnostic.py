import pandas as pd
import matplotlib.pyplot as plt

history_file = "results/subj01/stage2_informed_vit_vae/history.csv"
df = pd.read_csv(history_file)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Stage 2 Flow Matching Diagnostic (ViT VAE)")

# 1. Losses
ax = axes[0, 0]
ax.plot(df['epoch'], df['train_loss'], label='Train Total Loss')
ax.plot(df['epoch'], df['reg_loss'], label='Train Reg Loss', linestyle='--')
ax.plot(df['epoch'], df['flow_loss'], label='Train Flow Loss', linestyle='--')
ax.plot(df['epoch'], df['val_reg_loss'], label='Val Reg Loss', linestyle=':')
ax.plot(df['epoch'], df['val_flow_loss'], label='Val Flow Loss', linestyle=':')
ax.set_title("Losses")
ax.legend()

# 2. Latent Statistics
ax = axes[0, 1]
ax.plot(df['epoch'], df['val_reg_latent_mse'], label='Val Reg MSE')
ax.plot(df['epoch'], df['val_latent_mse'], label='Val Flow MSE')
ax.plot(df['epoch'], df['val_zgen_std'], label='Val Gen Std')
ax.set_title("Latent MSE & Std")
ax.legend()

# 3. fMRI PCC Output
ax = axes[1, 0]
ax.plot(df['epoch'], df['val_fmri_pcc'], label='Val fMRI PCC')
ax.plot(df['epoch'], df['val_fmri_spcc'], label='Val fMRI sPCC')
ax.set_title("fMRI Reconstruction Quality")
ax.legend()

# 4. Latent Cross Variance Ratio
ax = axes[1, 1]
ax.plot(df['epoch'], df['val_zgen_crossvar_ratio'], label='Z_gen CrossVar Ratio')
ax.plot(df['epoch'], df['val_reg_latent_pcc'], label='Val Reg PCC')
ax.plot(df['epoch'], df['val_latent_pcc'], label='Val Flow PCC')
ax.set_title("Diversity & Correlation")
ax.axhline(1.0, color='r', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("results/subj01/stage2_informed_vit_vae/diagnostic.png")
print("Saved diagnostic plot to results/subj01/stage2_informed_vit_vae/diagnostic.png")
