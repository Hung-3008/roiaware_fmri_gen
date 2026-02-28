"""
Extract VAE latents from trained Stage 1 fMRI MLP VAE.

Encodes fMRI data through the frozen VAE encoder to produce
latent codes (latent_dim,) for Stage 2 training.

Data mapping (follows data_notes.md conventions):
    fMRI source:   Data/nsd/subj0{sub}/nsd_{split}_fmri_zscore_sub{N}.npy
    Shape:         (N_images, 3, 15724)  — 3 repetitions per stimulus

    Extraction modes:
        'average'       → average 3 reps first, then encode → N_images latents
        'single_trial'  → encode each rep separately → 3 * N_images latents

    Output:
        {output_dir}/nsd_mlpvae_latents_{mode}_{split}_sub{N}.npy   (M, latent_dim)
        {output_dir}/nsd_mlpvae_latents_{mode}_{split}_sub{N}_index.npy   (M,) int32
            — maps each latent to original image index (0-based)
            — for 'average': index = [0, 1, 2, ..., N-1]
            — for 'single_trial': index = [0, 0, 0, 1, 1, 1, ..., N-1, N-1, N-1]

    Index alignment:
        latents[i] was encoded from stimulus image_index[i]
        image_index[i] corresponds to:
            fmri_raw[image_index[i]]     — original fMRI
            {split}_img/{image_index[i]}.png   — stimulus image
            clip_features[image_index[i]]       — CLIP features
            dinov2_features[image_index[i]]     — DINOv2 features

Usage:
    python -m src.extract_vae_latents \\
        --vae_ckpt results/fmri_mlp_vae/best.pt \\
        --config src/configs/fmri_mlp_vae.yaml

    python -m src.extract_vae_latents \\
        --vae_ckpt results/fmri_mlp_vae/best.pt \\
        --config src/configs/fmri_mlp_vae.yaml \\
        --rep_mode average --batch_size 256
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.model.fmri_mlp_vae import create_fmri_mlp_vae


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MLP VAE latents")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to MLP VAE checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True,
                        help="VAE config YAML (same used for training)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory (default: same as checkpoint dir / latents)")
    parser.add_argument("--rep_mode", type=str, default="average",
                        choices=["average", "single_trial"],
                        help="How to handle 3 repetitions (default: average)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def load_fmri(fmri_path: str, rep_mode: str, split: str):
    """
    Load fMRI data and build image index mapping.

    Args:
        fmri_path: path to .npy file, shape (N, 3, V)
        rep_mode: 'average' or 'single_trial'
        split: 'train' or 'test' (for logging)

    Returns:
        fmri: np.ndarray of shape (M, V) — flattened samples
        image_index: np.ndarray of shape (M,) int32
            — maps each sample to original image index (0-based)
    """
    raw = np.load(fmri_path, mmap_mode='r')
    print(f"  [{split}] Raw shape: {raw.shape}, dtype: {raw.dtype}")

    if raw.ndim == 3:  # (N, 3, V)
        n_images, n_reps, n_voxels = raw.shape

        if rep_mode == "average":
            # Average 3 reps → (N, V), 1-to-1 with images
            fmri = np.mean(raw, axis=1).astype(np.float32)
            image_index = np.arange(n_images, dtype=np.int32)
            print(f"  [{split}] Averaged {n_reps} reps → ({fmri.shape[0]}, {fmri.shape[1]})")

        elif rep_mode == "single_trial":
            # Each rep as separate sample → (3N, V)
            # image_index maps each trial back to original image
            # Layout: [img0_rep0, img0_rep1, img0_rep2, img1_rep0, img1_rep1, ...]
            fmri = raw.reshape(-1, n_voxels).astype(np.float32)
            image_index = np.repeat(np.arange(n_images, dtype=np.int32), n_reps)
            print(f"  [{split}] Single trial: {n_reps} reps × {n_images} images → "
                  f"({fmri.shape[0]}, {fmri.shape[1]})")
        else:
            raise ValueError(f"Unknown rep_mode: {rep_mode}")

    elif raw.ndim == 2:  # (N, V) — already flat
        fmri = raw.astype(np.float32)
        image_index = np.arange(raw.shape[0], dtype=np.int32)
        print(f"  [{split}] Already flat: {fmri.shape}")

    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {raw.shape}")

    return fmri, image_index


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load config ──
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # ── Output directory ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.vae_ckpt).parent / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load MLP VAE ──
    print(f"Loading MLP VAE from {args.vae_ckpt}...")
    model = create_fmri_mlp_vae(**model_cfg)

    ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    params = model.param_count()
    print(f"MLP VAE loaded: {params['total']:,} params ({params['total_mb']:.1f} MB)")
    print(f"Latent dim: {model_cfg.get('latent_dim', 1024)}")

    # ── Data paths (following data_notes.md) ──
    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))  # e.g. 1
    data_root = Path(data_cfg["root"])

    rep_mode = args.rep_mode
    print(f"\nSubject: {subject} (sub_num={sub_num})")
    print(f"Rep mode: {rep_mode}")
    print(f"Output dir: {output_dir}")

    # ── Extract for each split ──
    for split in ["train", "test"]:
        # NSD naming convention: Data/nsd/subj01/nsd_{split}_fmri_zscore_sub{N}.npy
        fmri_path = str(data_root / subject / f"nsd_{split}_fmri_zscore_sub{sub_num}.npy")
        print(f"\n{'='*60}")
        print(f"Extracting latents for [{split}]")
        print(f"  Source: {fmri_path}")

        if not Path(fmri_path).exists():
            print(f"  ⚠ File not found, skipping...")
            continue

        # Load fMRI with index mapping
        fmri, image_index = load_fmri(fmri_path, rep_mode, split)
        n_samples = fmri.shape[0]

        # Create DataLoader
        fmri_tensor = torch.from_numpy(fmri)
        dataset = TensorDataset(fmri_tensor)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        # ── Encode ──
        all_latents = []
        with torch.no_grad():
            for (batch_fmri,) in tqdm(loader, desc=f"Encoding {split}"):
                batch_fmri = batch_fmri.to(device)
                # Deterministic encoding: z = mu (no sampling)
                z, mu, logvar = model.encode(batch_fmri, sample_posterior=False)
                all_latents.append(mu.cpu())

        all_latents = torch.cat(all_latents, dim=0).numpy()  # (M, latent_dim)

        # ── Sanity checks ──
        assert all_latents.shape[0] == n_samples, \
            f"Mismatch: {all_latents.shape[0]} latents vs {n_samples} fMRI samples"
        assert all_latents.shape[0] == image_index.shape[0], \
            f"Mismatch: {all_latents.shape[0]} latents vs {image_index.shape[0]} indices"

        # ── Save latents + index ──
        mode_tag = "avg" if rep_mode == "average" else "single"
        latent_fname = f"nsd_mlpvae_latents_{mode_tag}_{split}_sub{sub_num}.npy"
        index_fname = f"nsd_mlpvae_latents_{mode_tag}_{split}_sub{sub_num}_index.npy"

        latent_path = output_dir / latent_fname
        index_path = output_dir / index_fname

        np.save(latent_path, all_latents.astype(np.float32))
        np.save(index_path, image_index)

        print(f"  ✅ Latents: {latent_path}")
        print(f"     Shape: {all_latents.shape}, Dtype: float32")
        print(f"     Stats: mean={all_latents.mean():.4f}, std={all_latents.std():.4f}, "
              f"min={all_latents.min():.4f}, max={all_latents.max():.4f}")
        print(f"  ✅ Index:   {index_path}")
        print(f"     Shape: {image_index.shape}, Range: [{image_index.min()}, {image_index.max()}]")

        # Verify mapping
        if rep_mode == "single_trial" and fmri.shape[0] > 0:
            unique_images = np.unique(image_index)
            counts = np.bincount(image_index)
            print(f"     Unique images: {len(unique_images)}, "
                  f"Reps per image: {counts[counts > 0].mean():.1f}")

        del all_latents, fmri, fmri_tensor, image_index

    print(f"\n{'='*60}")
    print(f"Done! All latents saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
