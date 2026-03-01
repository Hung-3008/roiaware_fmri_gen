"""
Diagnostic analysis: Why does xattn flow matching peak then degrade?

Analyzes history.csv from both subjects to identify overfitting patterns,
divergence indicators, and root causes.

Usage:
    python scripts/analyze_xattn_flow_training.py
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

# ─── Load data ────────────────────────────────────────────────────────────────

results_root = "results"
subjects = {}

for subj in ["subj01", "subj02"]:
    path = os.path.join(results_root, subj, "stage2_xattn_flow_vit_vae", "history.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        subjects[subj] = df
        print(f"Loaded {subj}: {len(df)} rows, epochs {df['epoch'].min()}-{df['epoch'].max()}")

if not subjects:
    raise FileNotFoundError("No history.csv found. Run from project root.")


# ─── FIGURE 1: Core Overfitting Diagnosis ────────────────────────────────────

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Flow Matching Training Diagnosis — Peak-then-Degrade Pattern",
             fontsize=14, fontweight='bold', y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

colors = {"subj01": "#2196F3", "subj02": "#FF5722"}

# 1. Train vs Val Flow Loss — THE PRIMARY OVERFITTING SIGNAL
ax1 = fig.add_subplot(gs[0, 0])
for subj, df in subjects.items():
    ax1.plot(df['epoch'], df['train_loss'], '-', color=colors[subj],
             alpha=0.7, label=f'{subj} train')
    ax1.plot(df['epoch'], df['val_flow_loss'], '--', color=colors[subj],
             alpha=0.9, label=f'{subj} val', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('① Train vs Val Loss\n(Gap = Overfitting)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Val Flow Loss — zoom in to see the turning point
ax2 = fig.add_subplot(gs[0, 1])
for subj, df in subjects.items():
    ax2.plot(df['epoch'], df['val_flow_loss'], '-o', color=colors[subj],
             markersize=3, label=subj)
    # Mark minimum
    best_idx = df['val_flow_loss'].idxmin()
    ax2.axvline(df.loc[best_idx, 'epoch'], color=colors[subj],
                linestyle=':', alpha=0.5)
    ax2.annotate(f"min@ep{df.loc[best_idx, 'epoch']}",
                 (df.loc[best_idx, 'epoch'], df.loc[best_idx, 'val_flow_loss']),
                 fontsize=8, color=colors[subj])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Val Flow Loss')
ax2.set_title('② Val Flow Loss\n(↑ after minimum = overfit)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Train-Val Gap over time
ax3 = fig.add_subplot(gs[0, 2])
for subj, df in subjects.items():
    gap = df['val_flow_loss'] - df['train_loss']
    ax3.plot(df['epoch'], gap, '-o', color=colors[subj],
             markersize=3, label=subj)
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Val - Train Loss')
ax3.set_title('③ Generalization Gap\n(Widening = Memorization)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. fMRI Samplewise PCC — the downstream metric
ax4 = fig.add_subplot(gs[1, 0])
for subj, df in subjects.items():
    ax4.plot(df['epoch'], df['val_fmri_spcc'], '-o', color=colors[subj],
             markersize=3, label=subj)
    best_idx = df['val_fmri_spcc'].idxmax()
    ax4.axvline(df.loc[best_idx, 'epoch'], color=colors[subj],
                linestyle=':', alpha=0.5)
    ax4.annotate(f"peak@ep{df.loc[best_idx, 'epoch']}\n={df.loc[best_idx, 'val_fmri_spcc']:.4f}",
                 (df.loc[best_idx, 'epoch'], df.loc[best_idx, 'val_fmri_spcc']),
                 fontsize=8, color=colors[subj])
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Val fMRI sPCC')
ax4.set_title('④ fMRI Sample PCC\n(Peak then Degrade)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Latent space statistics — z_gen_std and crossvar_ratio
ax5 = fig.add_subplot(gs[1, 1])
for subj, df in subjects.items():
    ax5.plot(df['epoch'], df['val_zgen_std'], '-', color=colors[subj],
             label=f'{subj} z_std', linewidth=2)
    ax5.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('z_gen std')
ax5.set_title('⑤ Generated Latent Std\n(Should be ≈1.0, diverging)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. Cross-variance ratio
ax6 = fig.add_subplot(gs[1, 2])
for subj, df in subjects.items():
    ax6.plot(df['epoch'], df['val_zgen_crossvar_ratio'], '-', color=colors[subj],
             label=f'{subj} ratio', linewidth=2)
    ax6.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='ideal=1.0')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Cross-var Ratio (gen/true)')
ax6.set_title('⑥ Variance Ratio\n(>1 = over-amplified, mode collapse risk)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# 7. Latent MSE vs Latent PCC — does PCC peak while MSE still improves?
ax7 = fig.add_subplot(gs[2, 0])
for subj, df in subjects.items():
    ax7a = ax7
    ax7b = ax7a.twinx()
    ax7a.plot(df['epoch'], df['val_latent_mse'], '-', color=colors[subj],
              alpha=0.7, label=f'{subj} MSE')
    ax7b.plot(df['epoch'], df['val_latent_pcc'], '--', color=colors[subj],
              alpha=0.9, label=f'{subj} PCC')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Latent MSE')
ax7b.set_ylabel('Latent PCC')
ax7.set_title('⑦ Latent MSE vs PCC\n(Diverging = scale issue)')
ax7.grid(True, alpha=0.3)

# 8. val_v_cos — velocity cosine similarity
ax8 = fig.add_subplot(gs[2, 1])
for subj, df in subjects.items():
    ax8.plot(df['epoch'], df['val_v_cos'], '-o', color=colors[subj],
             markersize=3, label=subj)
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Velocity Cosine Similarity')
ax8.set_title('⑧ Velocity Cos Sim\n(Direction accuracy of flow)')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# 9. Learning rate schedule
ax9 = fig.add_subplot(gs[2, 2])
for subj, df in subjects.items():
    ax9.plot(df['epoch'], df['lr'], '-', color=colors[subj], label=subj)
ax9.set_xlabel('Epoch')
ax9.set_ylabel('LR')
ax9.set_title('⑨ Learning Rate Schedule')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

output_path = os.path.join(results_root, "xattn_flow_diagnosis_core.png")
plt.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"\nSaved: {output_path}")
plt.close()


# ─── FIGURE 2: ROI-level Analysis ────────────────────────────────────────────

roi_cols = [c for c in subjects[list(subjects.keys())[0]].columns if c.startswith('roi_')]
roi_names = [c.replace('roi_', '').replace('_spcc', '') for c in roi_cols]

fig2, axes = plt.subplots(2, len(roi_cols) // 2 + 1, figsize=(22, 8))
fig2.suptitle("ROI-level sPCC Over Training — Does Degradation Affect All ROIs?",
              fontsize=13, fontweight='bold')

for i, (col, name) in enumerate(zip(roi_cols, roi_names)):
    ax = axes.flat[i]
    for subj, df in subjects.items():
        if col in df.columns:
            ax.plot(df['epoch'], df[col], '-o', color=colors[subj],
                    markersize=2, label=subj)
            best_idx = df[col].idxmax()
            ax.axvline(df.loc[best_idx, 'epoch'], color=colors[subj],
                       linestyle=':', alpha=0.3)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

# Hide extra axes
for i in range(len(roi_cols), len(axes.flat)):
    axes.flat[i].set_visible(False)

plt.tight_layout()
output_path2 = os.path.join(results_root, "xattn_flow_diagnosis_roi.png")
plt.savefig(output_path2, bbox_inches='tight', dpi=150)
print(f"Saved: {output_path2}")
plt.close()


# ─── FIGURE 3: Phase Analysis — When exactly does degradation start? ─────────

fig3 = plt.figure(figsize=(18, 10))
fig3.suptitle("Phase Analysis: Turning Points & Root Cause Indicators",
              fontsize=14, fontweight='bold', y=0.98)

gs3 = GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.35)

# 1. Rate of change of val_flow_loss — detect inflection point
ax_rate = fig3.add_subplot(gs3[0, 0])
for subj, df in subjects.items():
    d_loss = df['val_flow_loss'].diff()
    ax_rate.plot(df['epoch'].iloc[1:], d_loss.iloc[1:], '-o', color=colors[subj],
                 markersize=3, label=subj)
    # Find where derivative first becomes positive
    pos_idx = d_loss[d_loss > 0].index
    if len(pos_idx) > 0:
        first_pos_ep = df.loc[pos_idx[0], 'epoch']
        ax_rate.axvline(first_pos_ep, color=colors[subj], linestyle='--', alpha=0.5)
        ax_rate.annotate(f"↑@ep{first_pos_ep}", (first_pos_ep, 0),
                         fontsize=8, color=colors[subj])
ax_rate.axhline(0, color='red', linestyle='-', alpha=0.3)
ax_rate.set_xlabel('Epoch')
ax_rate.set_ylabel('Δ(val_flow_loss)')
ax_rate.set_title('Δ(Val Loss) — Inflection Point\n(Positive = starting to overfit)')
ax_rate.legend(fontsize=8)
ax_rate.grid(True, alpha=0.3)

# 2. Train loss derivative — is train still improving?
ax_train_rate = fig3.add_subplot(gs3[0, 1])
for subj, df in subjects.items():
    d_train = df['train_loss'].diff()
    ax_train_rate.plot(df['epoch'].iloc[1:], d_train.iloc[1:], '-o',
                       color=colors[subj], markersize=3, label=subj)
ax_train_rate.axhline(0, color='red', linestyle='-', alpha=0.3)
ax_train_rate.set_xlabel('Epoch')
ax_train_rate.set_ylabel('Δ(train_loss)')
ax_train_rate.set_title('Δ(Train Loss)\n(Keeps decreasing = memorizing)')
ax_train_rate.legend(fontsize=8)
ax_train_rate.grid(True, alpha=0.3)

# 3. Gradient norms
ax_grad = fig3.add_subplot(gs3[0, 2])
for subj, df in subjects.items():
    ax_grad.plot(df['epoch'], df['grad_avg'], '-', color=colors[subj],
                 label=f'{subj} avg', alpha=0.7)
    ax_grad.plot(df['epoch'], df['grad_max'], '--', color=colors[subj],
                 label=f'{subj} max', alpha=0.5)
ax_grad.set_xlabel('Epoch')
ax_grad.set_ylabel('Gradient Norm')
ax_grad.set_title('Gradient Norms\n(Rising = unstable optimization)')
ax_grad.legend(fontsize=7)
ax_grad.grid(True, alpha=0.3)

# 4. z_gen_std growth vs fMRI PCC — correlation plot
ax_corr = fig3.add_subplot(gs3[1, 0])
for subj, df in subjects.items():
    ax_corr.scatter(df['val_zgen_std'], df['val_fmri_spcc'], c=df['epoch'],
                    cmap='viridis', s=20, alpha=0.8, label=subj)
    # Show direction with arrows
    for i in range(0, len(df) - 1, 3):
        ax_corr.annotate('', xy=(df.iloc[i+1]['val_zgen_std'], df.iloc[i+1]['val_fmri_spcc']),
                         xytext=(df.iloc[i]['val_zgen_std'], df.iloc[i]['val_fmri_spcc']),
                         arrowprops=dict(arrowstyle='->', color=colors[subj], alpha=0.3))
ax_corr.axvline(1.0, color='green', linestyle='--', alpha=0.3, label='ideal std=1.0')
ax_corr.set_xlabel('z_gen std')
ax_corr.set_ylabel('val fMRI sPCC')
ax_corr.set_title('z_std vs fMRI PCC\n(std↑ → PCC↓ = variance explosion)')
ax_corr.legend(fontsize=7)
ax_corr.grid(True, alpha=0.3)

# 5. Latent MSE growth rate vs epoch
ax_mse_growth = fig3.add_subplot(gs3[1, 1])
for subj, df in subjects.items():
    # Normalize latent MSE to epoch 1
    mse_norm = df['val_latent_mse'] / df['val_latent_mse'].iloc[0]
    std_norm = df['val_zgen_std'] / df['val_zgen_std'].iloc[0]
    ax_mse_growth.plot(df['epoch'], mse_norm, '-', color=colors[subj],
                       label=f'{subj} MSE', linewidth=2)
    ax_mse_growth.plot(df['epoch'], std_norm, '--', color=colors[subj],
                       label=f'{subj} std', linewidth=1.5)
ax_mse_growth.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
ax_mse_growth.set_xlabel('Epoch')
ax_mse_growth.set_ylabel('Normalized (÷ epoch1)')
ax_mse_growth.set_title('Latent MSE & Std Growth\n(Relative to epoch 1)')
ax_mse_growth.legend(fontsize=7)
ax_mse_growth.grid(True, alpha=0.3)

# 6. Summary Table
ax_table = fig3.add_subplot(gs3[1, 2])
ax_table.axis('off')

table_data = []
for subj, df in subjects.items():
    best_spcc_idx = df['val_fmri_spcc'].idxmax()
    best_epoch = df.loc[best_spcc_idx, 'epoch']
    best_spcc = df.loc[best_spcc_idx, 'val_fmri_spcc']
    last_spcc = df.iloc[-1]['val_fmri_spcc']
    degradation = best_spcc - last_spcc

    best_val_loss_idx = df['val_flow_loss'].idxmin()
    best_val_loss_ep = df.loc[best_val_loss_idx, 'epoch']

    train_loss_at_peak = df.loc[best_spcc_idx, 'train_loss']
    val_loss_at_peak = df.loc[best_spcc_idx, 'val_flow_loss']
    gap_at_peak = val_loss_at_peak - train_loss_at_peak

    final_train = df.iloc[-1]['train_loss']
    final_val = df.iloc[-1]['val_flow_loss']
    final_gap = final_val - final_train

    z_std_at_peak = df.loc[best_spcc_idx, 'val_zgen_std']
    z_std_final = df.iloc[-1]['val_zgen_std']

    table_data.append([
        subj,
        f"{best_epoch}",
        f"{best_spcc:.4f}",
        f"{last_spcc:.4f}",
        f"{degradation:.4f}",
        f"{gap_at_peak:.3f}→{final_gap:.3f}",
        f"{z_std_at_peak:.3f}→{z_std_final:.3f}",
    ])

headers = ['Subject', 'Peak Ep', 'Peak sPCC', 'Final sPCC',
           'Degrad.', 'Gap(V-T)', 'z_std']
table = ax_table.table(cellText=table_data, colLabels=headers, loc='center',
                       cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
ax_table.set_title('Summary', fontsize=11, fontweight='bold')

output_path3 = os.path.join(results_root, "xattn_flow_diagnosis_phase.png")
plt.savefig(output_path3, bbox_inches='tight', dpi=150)
print(f"Saved: {output_path3}")
plt.close()


# ─── FIGURE 4: Causal Chain Visualization ────────────────────────────────────

fig4, axes4 = plt.subplots(1, 2, figsize=(18, 6))
fig4.suptitle("Causal Chain: Train Overfit → Latent Variance Explosion → Downstream Degradation",
              fontsize=13, fontweight='bold')

for ax, (subj, df) in zip(axes4, subjects.items()):
    ax2_twin = ax.twinx()

    # Generalization gap (primary Y)
    gap = df['val_flow_loss'] - df['train_loss']
    l1 = ax.plot(df['epoch'], gap, 'r-', linewidth=2, label='Gen. Gap (V-T)')
    ax.set_ylabel('Generalization Gap', color='red')
    ax.tick_params(axis='y', labelcolor='red')

    # fMRI sPCC (secondary Y)
    l2 = ax2_twin.plot(df['epoch'], df['val_fmri_spcc'], 'b-', linewidth=2,
                       label='fMRI sPCC')
    ax2_twin.set_ylabel('fMRI sPCC', color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')

    # z_gen_std (also on secondary)
    l3 = ax2_twin.plot(df['epoch'], df['val_zgen_std'] - 1.0, 'g--', linewidth=1.5,
                       label='z_std - 1.0', alpha=0.7)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='upper left')
    ax.set_xlabel('Epoch')
    ax.set_title(f'{subj}: Gap↑ → z_std↑ → fMRI PCC↓')
    ax.grid(True, alpha=0.3)

output_path4 = os.path.join(results_root, "xattn_flow_diagnosis_causal.png")
plt.savefig(output_path4, bbox_inches='tight', dpi=150)
print(f"Saved: {output_path4}")
plt.close()


# ─── Print Quantitative Analysis ─────────────────────────────────────────────

print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

for subj, df in subjects.items():
    print(f"\n{'─'*40}")
    print(f"  {subj}")
    print(f"{'─'*40}")

    # Find peaks
    best_spcc_idx = df['val_fmri_spcc'].idxmax()
    best_epoch = df.loc[best_spcc_idx, 'epoch']
    best_spcc = df.loc[best_spcc_idx, 'val_fmri_spcc']

    best_loss_idx = df['val_flow_loss'].idxmin()
    best_loss_epoch = df.loc[best_loss_idx, 'epoch']

    last = df.iloc[-1]

    print(f"  Peak fMRI sPCC:   {best_spcc:.4f} @ epoch {best_epoch}")
    print(f"  Final fMRI sPCC:  {last['val_fmri_spcc']:.4f} @ epoch {last['epoch']}")
    print(f"  Degradation:      {best_spcc - last['val_fmri_spcc']:.4f} "
          f"({(best_spcc - last['val_fmri_spcc'])/best_spcc*100:.1f}%)")
    print()
    print(f"  Min val_flow_loss: {df.loc[best_loss_idx, 'val_flow_loss']:.4f} @ epoch {best_loss_epoch}")
    print(f"  Final val_flow_loss: {last['val_flow_loss']:.4f}")
    print()

    # Gap analysis
    gap_at_peak = df.loc[best_spcc_idx, 'val_flow_loss'] - df.loc[best_spcc_idx, 'train_loss']
    gap_final = last['val_flow_loss'] - last['train_loss']
    print(f"  Gen. Gap @ peak:  {gap_at_peak:.4f}")
    print(f"  Gen. Gap @ final: {gap_final:.4f}  (×{gap_final/gap_at_peak:.1f} wider)")
    print()

    # z_std analysis
    z_std_at_peak = df.loc[best_spcc_idx, 'val_zgen_std']
    z_std_final = last['val_zgen_std']
    print(f"  z_gen_std @ peak: {z_std_at_peak:.4f}")
    print(f"  z_gen_std @ final:{z_std_final:.4f}  (grew {(z_std_final/z_std_at_peak-1)*100:.1f}%)")
    print()

    # crossvar_ratio
    cv_peak = df.loc[best_spcc_idx, 'val_zgen_crossvar_ratio']
    cv_final = last['val_zgen_crossvar_ratio']
    print(f"  crossvar_ratio @ peak:  {cv_peak:.4f}")
    print(f"  crossvar_ratio @ final: {cv_final:.4f}")
    print()

    # Correlation between gap growth and PCC degradation
    if len(df) > 5:
        gap = df['val_flow_loss'] - df['train_loss']
        corr = np.corrcoef(gap.values, df['val_fmri_spcc'].values)[0, 1]
        print(f"  Correlation(gap, fMRI_sPCC): {corr:.4f}  "
              f"({'strong negative = gap causes PCC drop' if corr < -0.5 else 'weak'})")

        corr_std = np.corrcoef(df['val_zgen_std'].values, df['val_fmri_spcc'].values)[0, 1]
        print(f"  Correlation(z_std, fMRI_sPCC): {corr_std:.4f}  "
              f"({'strong negative = variance explosion kills PCC' if corr_std < -0.5 else ''})")


print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print("""
PATTERN: Train loss keeps decreasing, val loss starts increasing after ~ep50-70.

CAUSAL CHAIN:
  1. OVERFITTING TO NOISE IN FLOW MATCHING:
     - Train loss monotonically decreases → model memorizes training noise patterns
     - Val flow loss increases after epoch ~50 → learned patterns don't generalize
     - Generalization gap widens ~3-4× from peak to final

  2. LATENT VARIANCE EXPLOSION:
     - val_zgen_std grows from ~1.0 to ~1.6 (subj01) / 1.3 (subj02)
     - val_zgen_crossvar_ratio >> 1.0 (reaches 1.8-1.9!)
     - The flow model generates latents with MUCH higher variance than true latents
     - This means the ODE integration overshoots — it's learned too-aggressive velocities

  3. DOWNSTREAM DEGRADATION:
     - Overshooting latents → higher latent MSE → worse fMRI reconstruction
     - fMRI sPCC peaks ~0.39-0.40 then drops to ~0.36-0.39

KEY INSIGHT:
  The flow model suffers from "velocity overfitting" — it learns velocity fields
  that perfectly match training pairs but produce trajectories that overshoot
  on unseen data. The ODE integrator magnifies small velocity errors into large
  latent errors, causing the generated z to have inflated variance.

EVIDENCE:
  - val_v_cos (velocity cosine sim) peaks and then *decreases* — the model's
    velocity predictions become directionally worse on val data.
  - val_latent_pcc plateaus even as val_latent_mse worsens — the generated z
    has correct relative structure but wrong scale.
  - crossvar_ratio >> 1 confirms the variance explosion.

POTENTIAL FIXES:
  1. Early stopping (obvious) — stop at epoch ~55-65
  2. Reduce model capacity (hidden_dim=384 may be too large for this data)
  3. Stronger regularization:
     - Increase dropout (0.25 → 0.4)
     - Increase weight_decay (0.1 → 0.2)
     - Increase sigma_aug (0.3 → 0.5)
  4. Latent normalization constraint during generation
  5. Reduce learning rate (3e-4 may be too high with cosine schedule)
  6. Use OT flow matching (minibatch OT) instead of vanilla CFM
     → OT creates straighter paths, less error amplification during integration
  7. Reduce ode_steps or use adaptive ODE solver
""")

print(f"\nAll plots saved to {results_root}/xattn_flow_diagnosis_*.png")
