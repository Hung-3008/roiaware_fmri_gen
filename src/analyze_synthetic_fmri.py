"""
analyze_synthetic_fmri.py
=========================
Compare synthetic fMRI (from Stage 2 FlowNP) against ground-truth fMRI.

Generates:
  - Distribution histograms (synthetic vs real)
  - Per-sample Pearson correlation
  - Voxel-wise statistics comparison
  - Summary text report

Usage:
  python src/analyze_synthetic_fmri.py \
    --synthetic_fmri Data/evals/flownp_compare/synthetic_fmri.npy \
    --json_file Data/evals/flownp_compare/test.json \
    --output_dir Data/evals/flownp_compare
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def pearson_corr_sample(pred, target):
    """Per-sample Pearson correlation."""
    corrs = []
    for i in range(pred.shape[0]):
        r, _ = stats.pearsonr(pred[i], target[i])
        corrs.append(r)
    return np.array(corrs)


def pearson_corr_voxelwise(pred, target):
    """Per-voxel Pearson correlation (across samples)."""
    n_voxels = pred.shape[1]
    corrs = []
    for v in range(n_voxels):
        if pred[:, v].std() < 1e-8 or target[:, v].std() < 1e-8:
            corrs.append(0.0)
        else:
            r, _ = stats.pearsonr(pred[:, v], target[:, v])
            corrs.append(r)
    return np.array(corrs)


def main():
    parser = argparse.ArgumentParser(description="Analyze synthetic vs real fMRI")
    parser.add_argument("--synthetic_fmri", type=str, required=True,
                        help="Path to synthetic_fmri.npy")
    parser.add_argument("--json_file", type=str, required=True,
                        help="JSON file with test_index entries")
    parser.add_argument("--real_fmri_path", type=str,
                        default="Data/nsd/subj01/nsd_test_fmri_zscore_sub1.npy",
                        help="Path to ground-truth test fMRI")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    syn_fmri = np.load(args.synthetic_fmri)  # (N, 15724)
    real_fmri_all = np.load(args.real_fmri_path)  # (1000, 15724) or (1000, 3, 15724)

    with open(args.json_file) as f:
        samples = json.load(f)

    # Handle multi-trial fMRI (average across trials)
    if real_fmri_all.ndim == 3:
        real_fmri_all = real_fmri_all.mean(axis=1).astype(np.float32)

    # Extract ground-truth for matching test indices
    test_indices = [s["test_index"] for s in samples]
    real_fmri = real_fmri_all[test_indices]  # (N, 15724)

    N = syn_fmri.shape[0]
    print(f"Samples: {N}")
    print(f"Synthetic shape: {syn_fmri.shape}, Real shape: {real_fmri.shape}")

    # ============================================================
    # 1. Basic Statistics
    # ============================================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  SYNTHETIC vs REAL fMRI ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Global stats
    report_lines.append("── Global Statistics ──")
    report_lines.append(f"{'Metric':<25} {'Synthetic':>15} {'Real':>15}")
    report_lines.append("-" * 55)
    for name, func in [("Mean", np.mean), ("Std", np.std),
                        ("Min", np.min), ("Max", np.max),
                        ("Median", np.median)]:
        report_lines.append(
            f"{name:<25} {func(syn_fmri):>15.6f} {func(real_fmri):>15.6f}")

    # Per-sample stats
    report_lines.append("")
    report_lines.append("── Per-sample Mean / Std ──")
    syn_means = syn_fmri.mean(axis=1)
    syn_stds = syn_fmri.std(axis=1)
    real_means = real_fmri.mean(axis=1)
    real_stds = real_fmri.std(axis=1)
    report_lines.append(f"{'Metric':<25} {'Synthetic':>15} {'Real':>15}")
    report_lines.append("-" * 55)
    report_lines.append(
        f"{'Mean of means':<25} {syn_means.mean():>15.6f} {real_means.mean():>15.6f}")
    report_lines.append(
        f"{'Std of means':<25} {syn_means.std():>15.6f} {real_means.std():>15.6f}")
    report_lines.append(
        f"{'Mean of stds':<25} {syn_stds.mean():>15.6f} {real_stds.mean():>15.6f}")
    report_lines.append(
        f"{'Std of stds':<25} {syn_stds.std():>15.6f} {real_stds.std():>15.6f}")

    # Amplitude range ratio
    syn_range = syn_fmri.max() - syn_fmri.min()
    real_range = real_fmri.max() - real_fmri.min()
    report_lines.append("")
    report_lines.append(f"Amplitude range (syn):  {syn_range:.4f}")
    report_lines.append(f"Amplitude range (real): {real_range:.4f}")
    report_lines.append(f"Range ratio (syn/real): {syn_range / real_range:.4f}")

    # ============================================================
    # 2. Per-sample Pearson Correlation
    # ============================================================
    sample_corrs = pearson_corr_sample(syn_fmri, real_fmri)
    report_lines.append("")
    report_lines.append("── Per-sample Pearson Correlation ──")
    report_lines.append(f"{'test_index':<15} {'PCC':>10}")
    report_lines.append("-" * 25)
    for i, s in enumerate(samples):
        report_lines.append(f"{s['test_index']:<15} {sample_corrs[i]:>10.4f}")
    report_lines.append("-" * 25)
    report_lines.append(
        f"{'Mean PCC':<15} {sample_corrs.mean():>10.4f}")
    report_lines.append(
        f"{'Std PCC':<15} {sample_corrs.std():>10.4f}")

    # ============================================================
    # 3. Per-sample MSE
    # ============================================================
    sample_mse = ((syn_fmri - real_fmri) ** 2).mean(axis=1)
    report_lines.append("")
    report_lines.append("── Per-sample MSE ──")
    report_lines.append(f"{'test_index':<15} {'MSE':>10}")
    report_lines.append("-" * 25)
    for i, s in enumerate(samples):
        report_lines.append(f"{s['test_index']:<15} {sample_mse[i]:>10.4f}")
    report_lines.append("-" * 25)
    report_lines.append(
        f"{'Mean MSE':<15} {sample_mse.mean():>10.4f}")

    # ============================================================
    # 4. KL Divergence / Distribution Comparison
    # ============================================================
    # Approximate KL via histogram
    bins = np.linspace(-4, 4, 200)
    syn_hist, _ = np.histogram(syn_fmri.flatten(), bins=bins, density=True)
    real_hist, _ = np.histogram(real_fmri.flatten(), bins=bins, density=True)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    syn_hist = syn_hist + eps
    real_hist = real_hist + eps
    kl_div = np.sum(real_hist * np.log(real_hist / syn_hist)) * (bins[1] - bins[0])

    report_lines.append("")
    report_lines.append("── Distribution Comparison ──")
    report_lines.append(f"KL Divergence (Real || Syn): {kl_div:.6f}")

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(
        syn_fmri.flatten()[:100000],
        real_fmri.flatten()[:100000])
    report_lines.append(f"KS Statistic: {ks_stat:.6f}, p-value: {ks_pval:.2e}")

    # Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = os.path.join(args.output_dir, "fmri_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved: {report_path}")

    # ============================================================
    # PLOTS
    # ============================================================

    # --- Fig 1: Global Distribution Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Synthetic vs Real fMRI Analysis", fontsize=14, fontweight='bold')

    # 1a: Overlaid histograms
    ax = axes[0, 0]
    ax.hist(real_fmri.flatten(), bins=150, alpha=0.6, density=True,
            label="Real", color="steelblue", range=(-4, 4))
    ax.hist(syn_fmri.flatten(), bins=150, alpha=0.6, density=True,
            label="Synthetic", color="coral", range=(-4, 4))
    ax.set_xlabel("Voxel Value")
    ax.set_ylabel("Density")
    ax.set_title("Value Distribution")
    ax.legend()

    # 1b: Per-sample mean comparison
    ax = axes[0, 1]
    x_pos = np.arange(N)
    width = 0.35
    ax.bar(x_pos - width/2, real_means, width, label="Real", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width/2, syn_means, width, label="Synthetic", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s["test_index"]) for s in samples], rotation=45, fontsize=8)
    ax.set_xlabel("Test Index")
    ax.set_ylabel("Mean Value")
    ax.set_title("Per-sample Mean")
    ax.legend()

    # 1c: Per-sample Std comparison
    ax = axes[1, 0]
    ax.bar(x_pos - width/2, real_stds, width, label="Real", color="steelblue", alpha=0.8)
    ax.bar(x_pos + width/2, syn_stds, width, label="Synthetic", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s["test_index"]) for s in samples], rotation=45, fontsize=8)
    ax.set_xlabel("Test Index")
    ax.set_ylabel("Std")
    ax.set_title("Per-sample Std")
    ax.legend()

    # 1d: Per-sample PCC bar
    ax = axes[1, 1]
    colors = ['green' if c > 0.3 else 'orange' if c > 0.1 else 'red'
              for c in sample_corrs]
    ax.bar(x_pos, sample_corrs, color=colors, alpha=0.8)
    ax.axhline(y=sample_corrs.mean(), color='black', linestyle='--',
               label=f"Mean PCC={sample_corrs.mean():.4f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s["test_index"]) for s in samples], rotation=45, fontsize=8)
    ax.set_xlabel("Test Index")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Per-sample PCC (Syn vs Real)")
    ax.legend()

    plt.tight_layout()
    fig1_path = os.path.join(args.output_dir, "fmri_analysis_overview.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Overview plot saved: {fig1_path}")

    # --- Fig 2: Per-sample scatter plots ---
    n_cols = 4
    n_rows = (N + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig2.suptitle("Per-sample: Synthetic vs Real fMRI (voxel scatter)",
                  fontsize=13, fontweight='bold')
    if n_rows == 1:
        axes2 = [axes2]

    for i, s in enumerate(samples):
        r, c = divmod(i, n_cols)
        ax = axes2[r][c] if n_rows > 1 else axes2[c]

        # Subsample voxels for scatter (too many to plot all)
        n_vox = syn_fmri.shape[1]
        idx = np.random.RandomState(42).choice(n_vox, min(2000, n_vox), replace=False)

        ax.scatter(real_fmri[i, idx], syn_fmri[i, idx], s=2, alpha=0.3, color="steelblue")
        lims = [min(real_fmri[i].min(), syn_fmri[i].min()),
                max(real_fmri[i].max(), syn_fmri[i].max())]
        ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.5)
        ax.set_xlabel("Real", fontsize=8)
        ax.set_ylabel("Synthetic", fontsize=8)
        ax.set_title(f"Test #{s['test_index']} (PCC={sample_corrs[i]:.3f})", fontsize=9)
        ax.set_aspect('equal', adjustable='box')

    # Hide empty axes
    for i in range(N, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes2[r][c] if n_rows > 1 else axes2[c]
        ax.axis('off')

    plt.tight_layout()
    fig2_path = os.path.join(args.output_dir, "fmri_scatter_per_sample.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Scatter plot saved: {fig2_path}")

    # --- Fig 3: Per-sample distribution overlay ---
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    fig3.suptitle("Per-sample Distribution: Synthetic vs Real",
                  fontsize=13, fontweight='bold')
    if n_rows == 1:
        axes3 = [axes3]

    for i, s in enumerate(samples):
        r, c = divmod(i, n_cols)
        ax = axes3[r][c] if n_rows > 1 else axes3[c]

        ax.hist(real_fmri[i], bins=80, alpha=0.5, density=True,
                label="Real", color="steelblue", range=(-3, 3))
        ax.hist(syn_fmri[i], bins=80, alpha=0.5, density=True,
                label="Syn", color="coral", range=(-3, 3))
        ax.set_title(f"Test #{s['test_index']}", fontsize=9)
        ax.legend(fontsize=7)

    for i in range(N, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes3[r][c] if n_rows > 1 else axes3[c]
        ax.axis('off')

    plt.tight_layout()
    fig3_path = os.path.join(args.output_dir, "fmri_dist_per_sample.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Distribution plot saved: {fig3_path}")

    print(f"\nAll analysis outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
