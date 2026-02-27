"""
save_images.py
==============
Consolidation of the original save_images.py + save_images_eval.py

Step 1: Extract PNG images from .npy stimuli arrays (for both train and test modes).
Step 2: Create evaluation tensor (all_images.pt) from test stimuli.

Input:
  - Data/nsd/subj0{sub}/nsd_train_stim_sub{sub}.npy  (N_train, 425, 425, 3) uint8
  - Data/nsd/subj0{sub}/nsd_test_stim_sub{sub}.npy   (N_test,  425, 425, 3) uint8

Output:
  - Data/nsd/subj0{sub}/train_img/{i}.png   (425x425 RGB PNG, i = 0..N_train-1)
  - Data/nsd/subj0{sub}/test_img/{i}.png    (425x425 RGB PNG, i = 0..N_test-1)
  - Data/evals/all_images.pt                (N_test, 3, 256, 256) float32 tensor

Usage:
  python SynBrain/data/save_images.py
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os

import argparse

# ==============================================================================
# Configuration
# ==============================================================================
parser = argparse.ArgumentParser(description="Extract PNG images and create evaluation tensor.")
parser.add_argument('--sub', type=int, default=1, help="Subject number (e.g., 1, 2, 5, 7)")
args = parser.parse_args()

sub = args.sub

# Paths (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../Data'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'nsd')
EVAL_DIR = os.path.join(BASE_DIR, 'evals')
os.makedirs(EVAL_DIR, exist_ok=True)

# ==============================================================================
# PART 1: Extract PNG images (original save_images.py logic)
# ==============================================================================
# Process both train and test modes
for mode in ['train', 'test']:
    npy_path = os.path.join(PROCESSED_DATA_DIR, f'subj0{sub}', f'nsd_{mode}_stim_sub{sub}.npy')

    if not os.path.exists(npy_path):
        print(f"Warning: {npy_path} not found. Skipping {mode} mode.")
        continue

    data = np.load(npy_path, mmap_mode='r')  # Memory-map to avoid loading ~5GB into RAM
    print(f"[{mode}] Data shape: {data.shape}")  # (N, 425, 425, 3)

    output_dir = os.path.join(PROCESSED_DATA_DIR, f'subj0{sub}', f'{mode}_img')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(data.shape[0]):
        img = Image.fromarray(data[i].astype(np.uint8))
        img.save(os.path.join(output_dir, f"{i}.png"))
        if (i + 1) % 500 == 0:
            print(f"  [{mode}] Saved {i+1}/{data.shape[0]} images...")

    print(f"[{mode}] All {data.shape[0]} images saved to: {output_dir}")

# ==============================================================================
# PART 2: Create evaluation tensor (original save_images_eval.py logic)
# ==============================================================================
# Only use test data for evaluation, matching the original save_images_eval.py
test_npy_path = os.path.join(PROCESSED_DATA_DIR, f'subj0{sub}', f'nsd_test_stim_sub{sub}.npy')

if os.path.exists(test_npy_path):
    data = np.load(test_npy_path, mmap_mode='r')  # Memory-map instead of full load

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    all_images = []
    for img in data:
        # (425, 425, 3) -> (3, 256, 256)
        img_tensor = transform(img)
        all_images.append(img_tensor)

    all_images_tensor = torch.stack(all_images, dim=0)

    save_path = os.path.join(EVAL_DIR, "all_images.pt")
    torch.save(all_images_tensor, save_path)
    print(f"Evaluation tensor shape: {all_images_tensor.shape}")  # (N, 3, 256, 256)
    print(f"Saved to: {save_path}")
else:
    print(f"Warning: {test_npy_path} not found. Skipping evaluation tensor creation.")

print("\nDone.")