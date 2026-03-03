import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_history(csv_path, output_dir=None):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV
    print(f"Loading history from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure integer epochs
    epochs = df['epoch']

    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Plot 1: Losses ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, df['train_loss'], 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    if 'val_flow_loss' in df.columns:
        ax2.plot(epochs, df['val_flow_loss'], 'r-', label='Val Flow Loss')
    if 'val_fmri_mse' in df.columns:
        ax2.plot(epochs, df['val_fmri_mse'], 'r--', label='Val fMRI MSE')
    ax2.set_ylabel('Validation Losses', color='r')
    ax2.tick_params('y', colors='r')
    
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.title('Training and Validation Losses over Epochs')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'losses.png'), dpi=300)
    plt.close()

    # --- Plot 2: Correlation Metrics ---
    plt.figure(figsize=(10, 6))
    if 'val_fmri_pcc' in df.columns:
        plt.plot(epochs, df['val_fmri_pcc'], label='Val fMRI PCC')
    if 'val_fmri_spcc' in df.columns:
        plt.plot(epochs, df['val_fmri_spcc'], label='Val fMRI sPCC', linewidth=2, color='green')
    if 'val_v_cos' in df.columns:
        plt.plot(epochs, df['val_v_cos'], label='Val V Cosine Sim', linestyle='--')
        
    plt.xlabel('Epoch')
    plt.ylabel('Correlation / Similarity')
    plt.title('Validation Correlations (PCC/sPCC/CosSim)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300)
    plt.close()

    # --- Plot 3: ROI sPCC ---
    roi_cols = [col for col in df.columns if col.startswith('roi_') and col.endswith('_spcc')]
    if roi_cols:
        plt.figure(figsize=(12, 8))
        for col in roi_cols:
            roi_name = col.replace('roi_', '').replace('_spcc', '')
            plt.plot(epochs, df[col], label=roi_name)
            
        plt.xlabel('Epoch')
        plt.ylabel('ROI sPCC')
        plt.title('Validation Spatial PCC per ROI')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roi_spcc.png'), dpi=300)
        plt.close()

    # --- Plot 4: Learning Rates & Gradients ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Learning Rates
    if 'lr_enc' in df.columns:
        ax1.plot(epochs, df['lr_enc'], label='LR Encoder', color='blue')
    if 'lr_vel' in df.columns:
        ax1.plot(epochs, df['lr_vel'], label='LR Velocity', color='cyan')
    ax1.set_ylabel('Learning Rate')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Learning Rates and Gradients')
    
    # Gradients
    if 'grad_avg' in df.columns:
        ax2.plot(epochs, df['grad_avg'], label='Avg Grad', color='orange')
    if 'grad_max' in df.columns:
        ax2.plot(epochs, df['grad_max'], label='Max Grad', color='red', alpha=0.5)
    ax2.set_ylabel('Gradient Norm')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_grads.png'), dpi=300)
    plt.close()
    
    print(f"Analysis plots saved to: {output_dir}")
    print("\nSummary at last epoch:")
    last_row = df.iloc[-1]
    print(f"Epoch: {last_row['epoch']}")
    print(f"Train Loss: {last_row['train_loss']:.4f}")
    if 'val_fmri_spcc' in df.columns: print(f"Val fMRI sPCC: {last_row['val_fmri_spcc']:.4f}")
    if 'val_flow_loss' in df.columns: print(f"Val Flow Loss: {last_row['val_flow_loss']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training history')
    parser.add_argument('--csv', type=str, required=True, help='Path to history.csv')
    parser.add_argument('--out', type=str, default=None, help='Output directory for plots (defaults to csv directory)')
    args = parser.parse_args()
    
    plot_history(args.csv, args.out)

