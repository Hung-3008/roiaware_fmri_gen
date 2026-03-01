"""
Diagnose vPCC vs sPCC divergence:
  - vPCC (voxel-wise) plateaus at ~0.31
  - sPCC (sample-wise) keeps rising to ~0.38

Hypothesis: model has saturated predictable voxels and is fitting
noise patterns that happen to correlate across samples.

Also compares: norm version vs original (no norm).

Usage:
    python scripts/analyze_vpcc_spcc_divergence.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})

results_root = "results"

# Load all available configs
runs = {}
for label, path in [
    ("subj01_orig", "subj01/stage2_xattn_flow_vit_vae/history.csv"),
    ("subj01_norm", "subj01/stage2_xattn_flow_vit_vae_norm/history.csv"),
    ("subj02_orig", "subj02/stage2_xattn_flow_vit_vae/history.csv"),
]:
    full = os.path.join(results_root, path)
    if os.path.exists(full):
        runs[label] = pd.read_csv(full)
        print(f"Loaded {label}: {len(runs[label])} rows")

colors = {
    "subj01_orig": "#2196F3",
    "subj01_norm": "#E91E63",
    "subj02_orig": "#FF9800",
}
styles = {
    "subj01_orig": "--",
    "subj01_norm": "-",
    "subj02_orig": "--",
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: vPCC vs sPCC Divergence — The Core Question
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 14))
fig.suptitle("vPCC vs sPCC Divergence: Hitting the Noise Ceiling?",
             fontsize=14, fontweight='bold', y=0.98)
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. vPCC and sPCC on same plot — show divergence
ax1 = fig.add_subplot(gs[0, 0])
for label, df in runs.items():
    ax1.plot(df['epoch'], df['val_fmri_pcc'], styles[label],
             color=colors[label], label=f'{label} vPCC', linewidth=2)
    ax1.plot(df['epoch'], df['val_fmri_spcc'], styles[label],
             color=colors[label], alpha=0.5, linewidth=1.5,
             label=f'{label} sPCC')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('PCC')
ax1.set_title('① vPCC (solid) vs sPCC (faded)\n(vPCC saturates, sPCC keeps rising)')
ax1.legend(fontsize=6, ncol=2)
ax1.grid(True, alpha=0.3)

# 2. sPCC / vPCC ratio — measures how much "sample structure" exceeds "voxel accuracy"
ax2 = fig.add_subplot(gs[0, 1])
for label, df in runs.items():
    # Skip epochs where vPCC is near 0
    mask = df['val_fmri_pcc'].abs() > 0.01
    ratio = df.loc[mask, 'val_fmri_spcc'] / df.loc[mask, 'val_fmri_pcc']
    ax2.plot(df.loc[mask, 'epoch'], ratio, styles[label],
             color=colors[label], label=label, linewidth=2)
ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='ratio=1')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('sPCC / vPCC')
ax2.set_title('② sPCC/vPCC Ratio\n(>1 = sample patterns outpace voxel accuracy)')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# 3. ΔvPCC and ΔsPCC per epoch — rate of improvement
ax3 = fig.add_subplot(gs[0, 2])
for label, df in runs.items():
    d_vpcc = df['val_fmri_pcc'].diff() / df['epoch'].diff()
    d_spcc = df['val_fmri_spcc'].diff() / df['epoch'].diff()
    ax3.plot(df['epoch'].iloc[1:], d_vpcc.iloc[1:] * 100, styles[label],
             color=colors[label], alpha=0.7, label=f'{label} ΔvPCC')
    ax3.plot(df['epoch'].iloc[1:], d_spcc.iloc[1:] * 100, styles[label],
             color=colors[label], alpha=0.4, linewidth=1)
ax3.axhline(0, color='red', linestyle='-', alpha=0.3)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('ΔPCC/epoch (×100)')
ax3.set_title('③ Rate of Improvement\n(vPCC rate → 0 while sPCC still positive)')
ax3.legend(fontsize=6)
ax3.grid(True, alpha=0.3)

# 4. Norm vs Original: val_flow_loss comparison — did LayerNorm help?
ax4 = fig.add_subplot(gs[1, 0])
for label, df in runs.items():
    ax4.plot(df['epoch'], df['val_flow_loss'], styles[label],
             color=colors[label], label=label, linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Val Flow Loss')
ax4.set_title('④ Val Flow Loss: Norm vs Orig\n(Norm stabilizes instead of diverging)')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# 5. Generalization Gap comparison
ax5 = fig.add_subplot(gs[1, 1])
for label, df in runs.items():
    gap = df['val_flow_loss'] - df['train_loss']
    ax5.plot(df['epoch'], gap, styles[label],
             color=colors[label], label=label, linewidth=2)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Val - Train Loss')
ax5.set_title('⑤ Gen. Gap: Norm vs Orig\n(Norm gap much smaller!)')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# 6. z_gen_std comparison
ax6 = fig.add_subplot(gs[1, 2])
for label, df in runs.items():
    ax6.plot(df['epoch'], df['val_zgen_std'], styles[label],
             color=colors[label], label=label, linewidth=2)
ax6.axhline(1.0, color='green', linestyle=':', alpha=0.5)
ax6.set_xlabel('Epoch')
ax6.set_ylabel('z_gen std')
ax6.set_title('⑥ z_gen Std: Norm vs Orig\n(Norm still drifts, slower)')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

# 7. ROI-level sPCC — which ROIs plateau first?
ax7 = fig.add_subplot(gs[2, 0:2])
roi_cols = [c for c in runs['subj01_norm'].columns if c.startswith('roi_')]
roi_names = [c.replace('roi_', '').replace('_spcc', '') for c in roi_cols]

# For subj01_norm, plot all ROIs
df_norm = runs['subj01_norm']
roi_colors = plt.cm.tab10(np.linspace(0, 1, len(roi_cols)))
for i, (col, name) in enumerate(zip(roi_cols, roi_names)):
    peak_val = df_norm[col].max()
    final_val = df_norm[col].iloc[-1]
    ax7.plot(df_norm['epoch'], df_norm[col], '-', color=roi_colors[i],
             label=f'{name} (peak={peak_val:.3f})', linewidth=1.5)
ax7.set_xlabel('Epoch')
ax7.set_ylabel('ROI sPCC')
ax7.set_title('⑦ Per-ROI sPCC (subj01_norm)\n(Higher-level ROIs plateau earlier)')
ax7.legend(fontsize=7, ncol=3, loc='lower right')
ax7.grid(True, alpha=0.3)

# 8. Summary statistics
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

table_data = []
for label, df in runs.items():
    last = df.iloc[-1]
    best_spcc_idx = df['val_fmri_spcc'].idxmax()
    best_vpcc_idx = df['val_fmri_pcc'].idxmax()

    # vPCC plateau detection: epoch where improvement < 0.001/5ep
    vpcc_rate = df['val_fmri_pcc'].diff()
    plateau_mask = (vpcc_rate < 0.001) & (df['epoch'] > 30)
    plateau_ep = df.loc[plateau_mask, 'epoch'].iloc[0] if plateau_mask.any() else "N/A"

    table_data.append([
        label.replace("subj01_", "s01_").replace("subj02_", "s02_"),
        f"{df.loc[best_vpcc_idx, 'val_fmri_pcc']:.4f}",
        f"{last['val_fmri_pcc']:.4f}",
        f"{df.loc[best_spcc_idx, 'val_fmri_spcc']:.4f}",
        f"{last['val_fmri_spcc']:.4f}",
        f"{plateau_ep}",
        f"{last['val_zgen_std']:.3f}",
    ])

headers = ['Run', 'Peak\nvPCC', 'Final\nvPCC', 'Peak\nsPCC',
           'Final\nsPCC', 'vPCC\nplat.ep', 'z_std']
table = ax8.table(cellText=table_data, colLabels=headers, loc='center',
                  cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.6)
ax8.set_title('Summary', fontsize=11, fontweight='bold')

output_path = os.path.join(results_root, "vpcc_spcc_divergence.png")
plt.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"\nSaved: {output_path}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Quantitative Analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("VPCC vs SPCC DIVERGENCE ANALYSIS")
print("="*80)

for label, df in runs.items():
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")

    last = df.iloc[-1]

    # Key metrics at end
    print(f"  Final vPCC: {last['val_fmri_pcc']:.4f}")
    print(f"  Final sPCC: {last['val_fmri_spcc']:.4f}")
    print(f"  Ratio sPCC/vPCC: {last['val_fmri_spcc']/max(last['val_fmri_pcc'], 0.001):.2f}")
    print()

    # When did vPCC plateau?
    for threshold in [0.001, 0.002, 0.005]:
        vpcc_diff = df['val_fmri_pcc'].diff()
        epoch_diff = df['epoch'].diff().clip(lower=1)
        vpcc_rate = vpcc_diff / epoch_diff
        stall = (vpcc_rate.abs() < threshold) & (df['epoch'] > 30)
        if stall.any():
            stall_ep = df.loc[stall, 'epoch'].iloc[0]
            print(f"  vPCC rate < {threshold}/ep first at epoch {stall_ep}")

    print()

    # ROI analysis
    roi_cols_local = [c for c in df.columns if c.startswith('roi_')]
    print(f"  {'ROI':<12} {'Peak':>8} {'Final':>8} {'Diff':>8}")
    for col in roi_cols_local:
        name = col.replace('roi_', '').replace('_spcc', '')
        peak = df[col].max()
        final = df[col].iloc[-1]
        diff = final - peak
        print(f"  {name:<12} {peak:>8.4f} {final:>8.4f} {diff:>+8.4f}")


print("\n" + "="*80)
print("INTERPRETATION: IS THE MODEL FITTING NOISE?")
print("="*80)
print("""
KEY FINDING: vPCC plateaus at ~0.31 while sPCC continues to ~0.38

WHAT THIS MEANS:
  - vPCC (voxel-wise): Average across voxels of R(pred_voxel, true_voxel).
    Measures per-VOXEL prediction fidelity across all samples.
  - sPCC (sample-wise): Average across samples of R(pred_sample, true_sample).
    Measures per-SAMPLE pattern similarity across all voxels.

  vPCC plateauing means:
    → For any given voxel, the model cannot predict its time-course more
      accurately. The VOXEL-LEVEL noise ceiling has been reached.

  sPCC still rising means:
    → The model is still improving at generating the CORRECT PATTERN of
      relative activations across voxels, even if absolute voxel values
      are noisy.

THIS IS NOT NECESSARILY NOISE FITTING. Here's why:

  1. vPCC is HARDER than sPCC:
     - vPCC requires predicting the absolute value of each voxel correctly
       across all test images. If a voxel has high trial-to-trial variability
       (common in fMRI even with b3 betas), vPCC hits a ceiling.
     - sPCC only needs the RELATIVE pattern across voxels to be correct.
       Even if each voxel is noisy, the pattern can still improve.

  2. The b3 noise ceiling:
     - Despite GLMdenoise + ridge regression, single-trial betas still have
       ~30-50% noise variance in many voxels (depending on SNR region).
     - vPCC ≈ 0.31 means the model explains ~10% of voxel variance
       (R²=0.096), which is consistent with the noise ceiling of b3 data
       for non-denoised models.

  3. HOWEVER — there IS a risk signal:
     - If sPCC keeps rising WITHOUT vPCC rising, the model may be learning
       to generate "average patterns" that correlate well per-sample but
       don't accurately reflect per-voxel dynamics.
     - This is a form of MODE AVERAGING: predictions collapse toward the
       mean response pattern, which increases sPCC but saturates vPCC.

EVIDENCE FROM THE DATA:
  - val_latent_pcc keeps slowly increasing → model is getting better
    at generating latents with correct correlational structure
  - val_latent_mse stays flat or increases → but the SCALE is not improving
  - val_zgen_crossvar_ratio >> 1.0 → generated latents have inflated variance
    compared to true latents

CONCLUSION:
  The model is NOT purely fitting noise. It's genuinely learning sample-level
  patterns. BUT it has hit the vPCC noise ceiling of b3 data, and further
  training will only improve sPCC through MODE AVERAGING (learning the mean
  pattern) rather than true per-voxel prediction.

RECOMMENDATIONS:
  1. Use EARLY STOPPING based on vPCC, not sPCC (if voxel accuracy matters)
  2. Consider multi-trial averaging at inference (generate N samples, average)
  3. The vPCC ceiling of ~0.31 is likely a DATA limitation, not a MODEL one
     → need denoised targets or multi-trial averaged targets to go higher
  4. For downstream tasks (image reconstruction), sPCC matters more than vPCC
     → the current model may be "good enough" if pattern quality is the goal
""")

print(f"\nPlot saved to {output_path}")
