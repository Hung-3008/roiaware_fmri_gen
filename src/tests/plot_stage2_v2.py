import pandas as pd
import matplotlib.pyplot as plt

history_file = "results/subj01/stage2_informed_vit_vae/history.csv"
df = pd.read_csv(history_file)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Stage 2 Flow Matching Diagnostic (ViT VAE + Updated Regressor)", fontsize=16)

# 1. Losses
ax = axes[0, 0]
ax.plot(df['epoch'], df['train_loss'], label='Train Total Loss', color='blue')
ax.plot(df['epoch'], df['reg_loss'], label='Train Reg Loss', linestyle='--', color='green')
ax.plot(df['epoch'], df['flow_loss'], label='Train Flow Loss', linestyle='--', color='orange')
ax.plot(df['epoch'], df['val_reg_loss'], label='Val Reg Loss', linestyle=':', color='darkgreen', linewidth=2)
ax.plot(df['epoch'], df['val_flow_loss'], label='Val Flow Loss', linestyle=':', color='darkred', linewidth=2)
ax.set_title("Training vs Validation Losses (Overfitting Check)")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# 2. Latent Space Accuracy (PCC)
ax = axes[0, 1]
ax.plot(df['epoch'], df['val_reg_latent_pcc'], label='Val Reg Latent PCC', color='green')
ax.plot(df['epoch'], df['val_latent_pcc'], label='Val Flow Latent PCC', color='orange')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_title("Latent Space PCC (Generative Accuracy)")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# 3. fMRI Space Accuracy (PCC)
ax = axes[1, 0]
ax.plot(df['epoch'], df['val_fmri_pcc'], label='Val Voxelwise PCC', color='purple')
ax.plot(df['epoch'], df['val_fmri_spcc'], label='Val Samplewise PCC', color='magenta')
ax.axhline(0.3, color='gray', linestyle='--', alpha=0.5)
ax.set_title("Target fMRI Space (Reconstruction Quality)")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# 4. Latent Norm / Std
ax = axes[1, 1]
ax.plot(df['epoch'], df['val_zgen_std'], label='Gen Latent Std', color='brown')
ax.plot(df['epoch'], df['delta_std'], label='Train Delta Std (z - z_bar)', color='gold', linestyle='--')
ax.plot(df['epoch'], df['val_delta_std'], label='Val Delta Std', color='gold', linestyle=':')
ax.set_title("Latent Variance / Flow Path Length")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# 5. Flow Vector Field Alignment
ax = axes[2, 0]
ax.plot(df['epoch'], df['val_v_cos'], label='Val Vector Cosine Sim', color='teal')
ax.set_title("Flow Field Alignment (v_pred vs target)")
ax.set_xlabel("Epoch")
ax.axhline(0.6, color='gray', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(alpha=0.3)

# 6. Specific ROIs
ax = axes[2, 1]
rois_to_plot = ['roi_V1_spcc', 'roi_hV4_spcc', 'roi_face_spcc', 'roi_place_spcc']
colors = ['red', 'blue', 'orange', 'green']
for roi, c in zip(rois_to_plot, colors):
    ax.plot(df['epoch'], df[roi], label=roi.replace('_spcc', ''), color=c)
ax.set_title("fMRI Samplewise PCC per ROI")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/subj01/stage2_informed_vit_vae/diagnostic_v2.png")
print("Saved diagnostic plot to results/subj01/stage2_informed_vit_vae/diagnostic_v2.png")
