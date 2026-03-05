"""
Per-ROI PCA Transform for fMRI data.

Fits independent PCA per brain ROI, reducing dimensionality while
preserving ROI structure. The output is naturally ROI-sorted (each ROI's
PCs are contiguous), making it compatible with ROI-aware patching.

Usage:
    pca = ROIPCATransform(decomposer, variance_ratio=0.95)
    pca.fit(train_fmri)          # (N, 15724) numpy
    z = pca.transform(fmri)      # (N, 15724) → (N, total_pcs)
    fmri_hat = pca.inverse_transform(z)  # (N, total_pcs) → (N, 15724)
    pca.save("roi_pca.pkl")
    pca = ROIPCATransform.load("roi_pca.pkl")
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.utils.roi_utils import ROIDecomposer


class ROIPCATransform:
    """Per-ROI PCA for fMRI dimensionality reduction.

    Fits an independent PCA per ROI and concatenates the reduced
    representations. The output vector is [PC_roi0, PC_roi1, ...],
    inherently ROI-sorted.
    """

    def __init__(
        self,
        decomposer: ROIDecomposer,
        variance_ratio: float = 0.95,
        n_components_per_roi: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            decomposer: ROIDecomposer with ROI voxel assignments.
            variance_ratio: Fraction of variance to retain per ROI
                (used when n_components_per_roi is not set for a ROI).
            n_components_per_roi: Optional dict {roi_name: n_components}.
                Overrides variance_ratio for specified ROIs.
        """
        self.decomposer = decomposer
        self.variance_ratio = variance_ratio
        self.n_components_per_roi = n_components_per_roi or {}
        self.pcas: Dict[str, PCA] = {}
        self._fitted = False

        # Will be set after fit()
        self._dims_per_roi: List[int] = []
        self._total_dims: int = 0
        self._roi_names: List[str] = []
        self._roi_offsets: List[int] = []  # start index for each ROI in PCA space

        # GPU buffers (lazily created)
        self._components_gpu: Optional[List[torch.Tensor]] = None
        self._means_gpu: Optional[List[torch.Tensor]] = None
        self._roi_indices_gpu: Optional[List[torch.Tensor]] = None
        self._device: Optional[torch.device] = None

    def fit(self, fmri_data: np.ndarray) -> "ROIPCATransform":
        """Fit PCA per ROI on training data.

        Args:
            fmri_data: (N, n_voxels) training fMRI in original voxel space.

        Returns:
            self
        """
        assert fmri_data.ndim == 2
        n_samples, n_voxels = fmri_data.shape
        assert n_voxels == self.decomposer.n_voxels, \
            f"Expected {self.decomposer.n_voxels} voxels, got {n_voxels}"

        self._roi_names = []
        self._dims_per_roi = []
        self._roi_offsets = []
        offset = 0

        for roi in self.decomposer.rois:
            roi_data = fmri_data[:, roi.indices]  # (N, n_voxels_roi)

            # Determine n_components
            if roi.name in self.n_components_per_roi:
                n_comp = min(
                    self.n_components_per_roi[roi.name],
                    roi.n_voxels, n_samples
                )
                pca = PCA(n_components=n_comp, svd_solver='full')
            else:
                # Use variance_ratio but cap at min(n_samples, n_voxels)
                max_comp = min(n_samples, roi.n_voxels)
                pca = PCA(
                    n_components=min(self.variance_ratio, max_comp),
                    svd_solver='full',
                )

            pca.fit(roi_data)
            self.pcas[roi.name] = pca

            actual_n = pca.n_components_
            self._roi_names.append(roi.name)
            self._dims_per_roi.append(actual_n)
            self._roi_offsets.append(offset)
            offset += actual_n

        self._total_dims = offset
        self._fitted = True
        # Reset GPU buffers
        self._components_gpu = None
        self._means_gpu = None
        self._roi_indices_gpu = None
        return self

    @property
    def total_dims(self) -> int:
        """Total PCA dimensions (sum of per-ROI components)."""
        assert self._fitted, "Call fit() first"
        return self._total_dims

    @property
    def dims_per_roi(self) -> List[int]:
        """Number of PCA components per ROI."""
        assert self._fitted, "Call fit() first"
        return list(self._dims_per_roi)

    def summary(self) -> str:
        """Human-readable summary of PCA decomposition."""
        assert self._fitted, "Call fit() first"
        lines = [
            f"Per-ROI PCA ({self.decomposer.n_voxels} voxels "
            f"→ {self._total_dims} PCs):"
        ]
        for roi, pca, n_pc in zip(
            self.decomposer.rois, self.pcas.values(), self._dims_per_roi
        ):
            var = pca.explained_variance_ratio_.sum() * 100
            lines.append(
                f"  {roi.name:20s}: {roi.n_voxels:5d} → {n_pc:4d} PCs "
                f"({var:5.1f}% var)"
            )
        return "\n".join(lines)

    # ── Numpy Transform / Inverse ────────────────────────────────────────

    def transform(self, fmri: np.ndarray) -> np.ndarray:
        """Transform fMRI voxels → PCA space.

        Args:
            fmri: (N, n_voxels) or (n_voxels,) numpy array.

        Returns:
            (N, total_pcs) or (total_pcs,) numpy array.
        """
        assert self._fitted
        squeeze = fmri.ndim == 1
        if squeeze:
            fmri = fmri[np.newaxis, :]

        parts = []
        for roi, name in zip(self.decomposer.rois, self._roi_names):
            roi_data = fmri[:, roi.indices]
            parts.append(self.pcas[name].transform(roi_data))

        out = np.concatenate(parts, axis=1).astype(np.float32)
        return out[0] if squeeze else out

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Inverse transform PCA space → voxels.

        Args:
            z: (N, total_pcs) or (total_pcs,) numpy array.

        Returns:
            (N, n_voxels) or (n_voxels,) numpy array.
        """
        assert self._fitted
        squeeze = z.ndim == 1
        if squeeze:
            z = z[np.newaxis, :]

        fmri = np.zeros(
            (z.shape[0], self.decomposer.n_voxels), dtype=np.float32)

        for roi, name, offset, n_pc in zip(
            self.decomposer.rois, self._roi_names,
            self._roi_offsets, self._dims_per_roi,
        ):
            roi_pcs = z[:, offset:offset + n_pc]
            fmri[:, roi.indices] = self.pcas[name].inverse_transform(roi_pcs)

        return fmri[0] if squeeze else fmri

    # ── GPU Tensor Transform / Inverse ───────────────────────────────────

    def _ensure_gpu_buffers(self, device: torch.device):
        """Lazily create GPU buffers for fast tensor operations."""
        if self._components_gpu is not None and self._device == device:
            return

        self._components_gpu = []
        self._means_gpu = []
        self._roi_indices_gpu = []
        self._device = device

        for roi, name in zip(self.decomposer.rois, self._roi_names):
            pca = self.pcas[name]
            # components_: (n_components, n_voxels_roi)
            self._components_gpu.append(
                torch.tensor(
                    pca.components_, dtype=torch.float32, device=device))
            self._means_gpu.append(
                torch.tensor(
                    pca.mean_, dtype=torch.float32, device=device))
            self._roi_indices_gpu.append(
                torch.tensor(
                    roi.indices, dtype=torch.long, device=device))

    @torch.no_grad()
    def transform_torch(self, fmri: torch.Tensor) -> torch.Tensor:
        """Transform fMRI voxels → PCA space (GPU tensor).

        Args:
            fmri: (B, n_voxels) tensor on device.

        Returns:
            (B, total_pcs) tensor on same device.
        """
        assert self._fitted
        self._ensure_gpu_buffers(fmri.device)

        parts = []
        for indices, components, mean in zip(
            self._roi_indices_gpu, self._components_gpu, self._means_gpu,
        ):
            roi_data = fmri[:, indices] - mean.unsqueeze(0)  # (B, V_roi)
            # (B, V_roi) @ (V_roi, n_pc) = (B, n_pc)
            parts.append(roi_data @ components.T)

        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def inverse_transform_torch(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transform PCA space → voxels (GPU tensor).

        Args:
            z: (B, total_pcs) tensor on device.

        Returns:
            (B, n_voxels) tensor on same device.
        """
        assert self._fitted
        self._ensure_gpu_buffers(z.device)

        fmri = torch.zeros(
            z.shape[0], self.decomposer.n_voxels,
            device=z.device, dtype=z.dtype)

        for roi_idx, (indices, components, mean) in enumerate(zip(
            self._roi_indices_gpu, self._components_gpu, self._means_gpu,
        )):
            offset = self._roi_offsets[roi_idx]
            n_pc = self._dims_per_roi[roi_idx]
            roi_pcs = z[:, offset:offset + n_pc]  # (B, n_pc)
            # (B, n_pc) @ (n_pc, V_roi) + mean = (B, V_roi)
            fmri[:, indices] = roi_pcs @ components + mean.unsqueeze(0)

        return fmri

    # ── Patch ROI IDs ────────────────────────────────────────────────────

    def get_patch_roi_ids(self, patch_size: int) -> List[int]:
        """Compute patch_roi_ids for ROI-aware positional embedding.

        Since PCA output is [PC_roi0, PC_roi1, ...], each patch's ROI
        is determined by which ROI's PCs it covers.

        Args:
            patch_size: Number of PCA dims per patch.

        Returns:
            List of ROI indices, one per patch.
        """
        assert self._fitted
        # Expand ROI labels to per-dim
        dim_roi_labels = []
        for roi_idx, n_pc in enumerate(self._dims_per_roi):
            dim_roi_labels.extend([roi_idx] * n_pc)

        # Pad to multiple of patch_size
        total = len(dim_roi_labels)
        padded = ((total + patch_size - 1) // patch_size) * patch_size
        pad_len = padded - total
        # Pad with last ROI label (typically "other" = 8)
        last_roi = len(self._dims_per_roi) - 1
        dim_roi_labels.extend([last_roi] * pad_len)

        # Majority vote per patch
        num_patches = padded // patch_size
        patch_roi_ids = []
        for p in range(num_patches):
            chunk = dim_roi_labels[p * patch_size:(p + 1) * patch_size]
            counts = np.bincount(chunk, minlength=len(self.decomposer.rois))
            patch_roi_ids.append(int(np.argmax(counts)))

        return patch_roi_ids

    def get_roi_dim_ranges(self) -> List[torch.Tensor]:
        """Get PCA dimension indices for each ROI (for per-ROI soft target).

        Returns:
            List of LongTensor index arrays, one per ROI, indexing into
            the PCA-space vector.
        """
        assert self._fitted
        ranges = []
        for offset, n_pc in zip(self._roi_offsets, self._dims_per_roi):
            ranges.append(
                torch.arange(offset, offset + n_pc, dtype=torch.long))
        return ranges

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str):
        """Save PCA transform to disk."""
        assert self._fitted
        data = {
            "pcas": self.pcas,
            "variance_ratio": self.variance_ratio,
            "n_components_per_roi": self.n_components_per_roi,
            "dims_per_roi": self._dims_per_roi,
            "total_dims": self._total_dims,
            "roi_names": self._roi_names,
            "roi_offsets": self._roi_offsets,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, decomposer: ROIDecomposer) -> "ROIPCATransform":
        """Load PCA transform from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            decomposer,
            variance_ratio=data["variance_ratio"],
            n_components_per_roi=data.get("n_components_per_roi", {}),
        )
        obj.pcas = data["pcas"]
        obj._dims_per_roi = data["dims_per_roi"]
        obj._total_dims = data["total_dims"]
        obj._roi_names = data["roi_names"]
        obj._roi_offsets = data["roi_offsets"]
        obj._fitted = True
        return obj
