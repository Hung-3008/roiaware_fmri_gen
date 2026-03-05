"""
ROI Utilities for NSD fMRI data.

Provides ROI decomposition of nsdgeneral mask into 9 functional groups:
    - V1, V2, V3, hV4 (retinotopic, from prf-visualrois)
    - Body-selective, Face-selective, Place-selective, Word-selective (from floc-*)
    - Other visual (remaining nsdgeneral voxels)

Each group is assigned a hierarchy level for DINOv2 layer alignment:
    - Early (V1, V2): best predicted by early DNN layers
    - Mid (V3, hV4): best predicted by mid DNN layers
    - Late (floc-*): best predicted by late DNN layers
    - All (other): uses CLS or all layers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch


# ─── ROI Group Definitions ───────────────────────────────────────────────────

ROI_GROUPS = [
    "V1", "V2", "V3", "hV4",
    "body", "face", "place", "word",
    "other",
]

# Hierarchy level for each ROI group (for DINOv2 layer selection)
ROI_HIERARCHY = {
    "V1": "early",
    "V2": "early",
    "V3": "mid",
    "hV4": "mid",
    "body": "late",
    "face": "late",
    "place": "late",
    "word": "late",
    "other": "all",
}


# ─── ROI Decomposer ──────────────────────────────────────────────────────────


@dataclass
class ROIInfo:
    """Information about a single ROI group."""
    name: str
    n_voxels: int
    indices: np.ndarray        # indices into the flat nsdgeneral array
    hierarchy: str             # "early", "mid", "late", "all"


class ROIDecomposer:
    """
    Decomposes nsdgeneral fMRI vectors into 9 ROI groups.

    Usage:
        decomposer = ROIDecomposer(roi_dir)
        roi_vectors = decomposer.split(fmri_flat)     # dict of (B, n_voxels_roi)
        fmri_flat = decomposer.assemble(roi_vectors)   # (B, 15724)
    """

    def __init__(self, roi_dir: str):
        """
        Args:
            roi_dir: Path to NSD ROI directory,
                     e.g. "Data/nsddata/ppdata/subj01/func1pt8mm/roi"
        """
        self.roi_dir = roi_dir
        self.rois: List[ROIInfo] = []
        self._build_roi_mapping()

    def _build_roi_mapping(self):
        """Build ROI assignment for each nsdgeneral voxel."""
        # Load nsdgeneral mask
        nsdgen = nib.load(f"{self.roi_dir}/nsdgeneral.nii.gz").get_fdata()
        nsdgen_mask = nsdgen > 0
        nsdgen_flat = nsdgen_mask.flatten()
        self.nsdgen_indices = np.where(nsdgen_flat)[0]
        self.n_voxels = len(self.nsdgen_indices)

        # Load ROI masks
        prf = nib.load(
            f"{self.roi_dir}/prf-visualrois.nii.gz").get_fdata().flatten()
        floc_bodies = nib.load(
            f"{self.roi_dir}/floc-bodies.nii.gz").get_fdata().flatten()
        floc_faces = nib.load(
            f"{self.roi_dir}/floc-faces.nii.gz").get_fdata().flatten()
        floc_places = nib.load(
            f"{self.roi_dir}/floc-places.nii.gz").get_fdata().flatten()
        floc_words = nib.load(
            f"{self.roi_dir}/floc-words.nii.gz").get_fdata().flatten()

        # Assign each nsdgeneral voxel to an ROI group
        # Priority: prf-visualrois > floc > "other"
        assignments = np.full(self.n_voxels, 8, dtype=np.int32)  # default: other

        for i, vox_idx in enumerate(self.nsdgen_indices):
            p = int(prf[vox_idx])
            if p in [1, 2]:       # V1v, V1d
                assignments[i] = 0
            elif p in [3, 4]:     # V2v, V2d
                assignments[i] = 1
            elif p in [5, 6]:     # V3v, V3d
                assignments[i] = 2
            elif p == 7:          # hV4
                assignments[i] = 3
            elif int(floc_faces[vox_idx]) > 0:
                assignments[i] = 5
            elif int(floc_bodies[vox_idx]) > 0:
                assignments[i] = 4
            elif int(floc_places[vox_idx]) > 0:
                assignments[i] = 6
            elif int(floc_words[vox_idx]) > 0:
                assignments[i] = 7

        self.assignments = assignments

        # Build ROIInfo for each group
        for group_idx, name in enumerate(ROI_GROUPS):
            mask = assignments == group_idx
            indices = np.where(mask)[0]
            self.rois.append(ROIInfo(
                name=name,
                n_voxels=len(indices),
                indices=indices,
                hierarchy=ROI_HIERARCHY[name],
            ))

    def split(self, fmri: torch.Tensor) -> List[torch.Tensor]:
        """
        Split flat fMRI vector into ROI groups.

        Args:
            fmri: (B, n_voxels) flat fMRI data

        Returns:
            List of (B, n_voxels_roi) tensors, one per ROI group
        """
        return [fmri[:, roi.indices] for roi in self.rois]

    def assemble(self, roi_vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Assemble ROI vectors back into flat fMRI vector.

        Args:
            roi_vectors: List of (B, n_voxels_roi) tensors

        Returns:
            (B, n_voxels) flat fMRI data
        """
        B = roi_vectors[0].shape[0]
        device = roi_vectors[0].device
        out = torch.zeros(B, self.n_voxels, device=device)
        for roi, vec in zip(self.rois, roi_vectors):
            out[:, roi.indices] = vec
        return out

    def get_roi_sizes(self) -> List[int]:
        """Return list of voxel counts per ROI group."""
        return [roi.n_voxels for roi in self.rois]

    def get_roi_names(self) -> List[str]:
        """Return list of ROI group names."""
        return [roi.name for roi in self.rois]

    def summary(self) -> str:
        """Print summary of ROI decomposition."""
        lines = [f"ROI Decomposition ({self.n_voxels} total voxels):"]
        for roi in self.rois:
            pct = 100 * roi.n_voxels / self.n_voxels
            lines.append(
                f"  {roi.name:20s}: {roi.n_voxels:5d} voxels "
                f"({pct:5.1f}%) [{roi.hierarchy}]")
        return "\n".join(lines)


# ─── ROI-Sort Utilities ───────────────────────────────────────────────────────


def build_roi_perm(decomposer: ROIDecomposer, patch_size: int, n_voxels: int):
    """
    Build permutation indices for ROI-sorted fMRI voxels and per-patch ROI labels.

    Rearranges the flat fMRI vector so that voxels in the same ROI are
    contiguous. After patchification, each patch will contain voxels
    predominantly from a single ROI, giving the model meaningful positional
    structure.

    Args:
        decomposer: ROIDecomposer with .assignments array (shape n_voxels,)
        patch_size: number of voxels per patch (e.g. 124)
        n_voxels: total number of voxels (e.g. 15724)

    Returns:
        voxel_perm:    np.ndarray int64 (n_voxels,) — permutation index
        patch_roi_ids: np.ndarray int64 (num_patches,) — dominant ROI per patch
    """
    # Build permutation: concatenate voxel indices grouped by ROI in order
    roi_order = [roi.indices for roi in decomposer.rois]
    perm = np.concatenate(roi_order).astype(np.int64)   # (n_voxels,)
    assert len(perm) == n_voxels, \
        f"Permutation length {len(perm)} != n_voxels {n_voxels}"

    # Build per-voxel ROI assignment in the NEW (sorted) order
    voxel_roi_sorted = decomposer.assignments[perm]  # (n_voxels,)

    # Pad to multiple of patch_size
    padded = ((n_voxels + patch_size - 1) // patch_size) * patch_size
    pad_len = padded - n_voxels
    if pad_len > 0:
        # Pad with label 8 ("other") for the dummy voxels
        voxel_roi_padded = np.concatenate([
            voxel_roi_sorted, np.full(pad_len, 8, dtype=np.int64)])
    else:
        voxel_roi_padded = voxel_roi_sorted

    num_patches = padded // patch_size

    # Majority-vote: for each patch, pick the most common ROI label
    patch_roi_ids = np.zeros(num_patches, dtype=np.int64)
    for p in range(num_patches):
        chunk = voxel_roi_padded[p * patch_size: (p + 1) * patch_size]
        roi_counts = np.bincount(chunk, minlength=len(ROI_GROUPS))
        patch_roi_ids[p] = int(np.argmax(roi_counts))

    return perm, patch_roi_ids

