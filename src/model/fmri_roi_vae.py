"""
fMRI ROI-level VAE — Stage 1.

Instead of treating voxels as an unordered flat vector, this architecture
decomposes fMRI into ROI groups (V1, V2, V3, hV4, body, face, place, word, other)
and processes each ROI with a dedicated MLP encoder/decoder.

Architecture:
    Encoder:
        fMRI (15724) → split by ROI → per-ROI MLP → ROI tokens (9 × embed_dim)
        → + ROI Embedding → Transformer Encoder → CLS token → μ, logvar (768)

    Decoder:
        z (768) → expand to 9 tokens → + ROI Embedding → Transformer Decoder
        → per-ROI MLP → assemble → fMRI_recon (15724)

Key advantages:
    - Each token has semantic meaning (= a brain region)
    - Only 9-10 tokens → self-attention is extremely efficient
    - Per-ROI MLPs handle variable voxel counts naturally
    - Attention learns cross-ROI interactions (V1↔V2, face↔place, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.roi_utils import ROIDecomposer


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class FmriROIVAEConfig:
    """Configuration for FmriROIVAE."""
    roi_dir: str = "Data/nsddata/ppdata/subj01/func1pt8mm/roi"
    embed_dim: int = 512         # Dimension of each ROI token
    depth: int = 4               # Number of Transformer layers
    heads: int = 8               # Attention heads
    mlp_ratio: float = 4.0       # MLP expansion ratio in Transformer
    latent_dim: int = 768        # Output latent dimension
    dropout: float = 0.1         # Dropout rate
    roi_mlp_hidden: int = 512    # Hidden dim for per-ROI MLPs
    roi_mlp_layers: int = 2      # Number of hidden layers in per-ROI MLPs
    n_tokens_per_large_roi: int = 1  # Use multiple tokens for large ROIs (other)
    large_roi_threshold: int = 3000  # ROIs with more voxels than this get extra tokens


# ─── Per-ROI MLP ─────────────────────────────────────────────────────────────


class ROIEncoder(nn.Module):
    """MLP that maps voxels of one ROI → one or more tokens."""

    def __init__(self, n_voxels: int, embed_dim: int, hidden_dim: int = 512,
                 n_layers: int = 2, n_tokens: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        out_dim = embed_dim * n_tokens

        layers = [nn.Linear(n_voxels, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.LayerNorm(hidden_dim),
                       nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_voxels) → (B, n_tokens, embed_dim)"""
        out = self.net(x)  # (B, embed_dim * n_tokens)
        if self.n_tokens > 1:
            return out.view(out.shape[0], self.n_tokens, -1)
        return out.unsqueeze(1)  # (B, 1, embed_dim)


class ROIDecoder(nn.Module):
    """MLP that maps one or more tokens → voxels of one ROI."""

    def __init__(self, n_voxels: int, embed_dim: int, hidden_dim: int = 512,
                 n_layers: int = 2, n_tokens: int = 1, dropout: float = 0.1):
        super().__init__()
        in_dim = embed_dim * n_tokens

        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.LayerNorm(hidden_dim),
                       nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, n_voxels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_tokens, embed_dim) → (B, n_voxels)"""
        B = x.shape[0]
        x = x.reshape(B, -1)  # flatten tokens: (B, n_tokens * embed_dim)
        return self.net(x)


# ─── Transformer Block ──────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ─── ROI-VAE Model ───────────────────────────────────────────────────────────


class FmriROIVAE(nn.Module):
    """
    ROI-level VAE for fMRI data.

    Decomposes fMRI into ROI groups, encodes each to a token,
    applies Transformer self-attention across ROI tokens,
    then produces a compact latent representation.

    Interface is compatible with FmriViTVAE / FmriMLPVAE:
        encode(x) → (z, mu, logvar)
        decode(z) → x_recon
        compute_loss(x, x_recon, mu, logvar, ...) → dict
    """

    def __init__(self, config: Optional[FmriROIVAEConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = FmriROIVAEConfig(**kwargs)
        self.config = config

        # ── ROI Decomposer ──
        self.decomposer = ROIDecomposer(config.roi_dir)
        self.n_voxels = self.decomposer.n_voxels
        roi_sizes = self.decomposer.get_roi_sizes()
        roi_names = self.decomposer.get_roi_names()
        n_rois = len(roi_sizes)

        # Determine number of tokens per ROI
        self.tokens_per_roi = []
        for i, size in enumerate(roi_sizes):
            if size > config.large_roi_threshold:
                self.tokens_per_roi.append(
                    max(1, size // config.large_roi_threshold))
            else:
                self.tokens_per_roi.append(1)
        self.total_tokens = sum(self.tokens_per_roi)

        # ── Per-ROI Encoders ──
        self.roi_encoders = nn.ModuleList()
        for i, size in enumerate(roi_sizes):
            self.roi_encoders.append(ROIEncoder(
                n_voxels=size,
                embed_dim=config.embed_dim,
                hidden_dim=config.roi_mlp_hidden,
                n_layers=config.roi_mlp_layers,
                n_tokens=self.tokens_per_roi[i],
                dropout=config.dropout,
            ))

        # ── CLS Token + ROI Embeddings ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        # Positional/ROI embedding for all tokens + CLS
        self.roi_embed = nn.Parameter(
            torch.randn(1, self.total_tokens + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)

        # ── Transformer Encoder ──
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.heads,
                             config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)

        # ── Latent Projection ──
        self.fc_mu = nn.Linear(config.embed_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.embed_dim, config.latent_dim)

        # ── Decoder ──
        # z → expand to total_tokens tokens
        self.dec_embed = nn.Linear(
            config.latent_dim, self.total_tokens * config.embed_dim)
        self.dec_roi_embed = nn.Parameter(
            torch.randn(1, self.total_tokens, config.embed_dim))
        self.dec_pos_drop = nn.Dropout(config.dropout)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.heads,
                             config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)

        # ── Per-ROI Decoders ──
        self.roi_decoders = nn.ModuleList()
        for i, size in enumerate(roi_sizes):
            self.roi_decoders.append(ROIDecoder(
                n_voxels=size,
                embed_dim=config.embed_dim,
                hidden_dim=config.roi_mlp_hidden,
                n_layers=config.roi_mlp_layers,
                n_tokens=self.tokens_per_roi[i],
                dropout=config.dropout,
            ))

        self._init_weights()

        # Log architecture
        print(f"FmriROIVAE Architecture:")
        print(f"  Total voxels: {self.n_voxels}")
        print(f"  ROIs: {n_rois}, Total tokens: {self.total_tokens} + 1 CLS")
        for i, name in enumerate(roi_names):
            print(f"    {name:>8s}: {roi_sizes[i]:5d} voxels → "
                  f"{self.tokens_per_roi[i]} token(s)")
        print(f"  Transformer: depth={config.depth}, dim={config.embed_dim}, "
              f"heads={config.heads}")
        print(f"  Latent dim: {config.latent_dim}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.roi_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_roi_embed, std=0.02)

    def encode(self, x: torch.Tensor,
               sample_posterior: bool = True
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode fMRI → latent.

        Args:
            x: (B, n_voxels)
            sample_posterior: If True, sample z ~ N(μ, σ²); else z = μ.

        Returns:
            z: (B, latent_dim)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        B = x.shape[0]

        # 1. Split by ROI
        roi_vectors = self.decomposer.split(x)

        # 2. Encode each ROI → tokens
        tokens = []
        for i, (roi_vec, encoder) in enumerate(
                zip(roi_vectors, self.roi_encoders)):
            tok = encoder(roi_vec)  # (B, n_tokens_i, embed_dim)
            tokens.append(tok)
        tokens = torch.cat(tokens, dim=1)  # (B, total_tokens, embed_dim)

        # 3. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, total_tokens+1, embed_dim)

        # 4. Add ROI embeddings
        tokens = tokens + self.roi_embed
        tokens = self.pos_drop(tokens)

        # 5. Transformer encoder
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        tokens = self.encoder_norm(tokens)

        # 6. CLS → latent
        cls_out = tokens[:, 0]  # (B, embed_dim)
        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent → fMRI.

        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, n_voxels)
        """
        B = z.shape[0]

        # 1. Project z to token sequence
        tokens = self.dec_embed(z)  # (B, total_tokens * embed_dim)
        tokens = tokens.view(B, self.total_tokens, self.config.embed_dim)

        # 2. Add ROI embeddings
        tokens = tokens + self.dec_roi_embed
        tokens = self.dec_pos_drop(tokens)

        # 3. Transformer decoder
        for blk in self.decoder_blocks:
            tokens = blk(tokens)
        tokens = self.decoder_norm(tokens)

        # 4. Split tokens back to per-ROI and decode
        roi_recons = []
        token_idx = 0
        for i, decoder in enumerate(self.roi_decoders):
            n_tok = self.tokens_per_roi[i]
            roi_tokens = tokens[:, token_idx:token_idx + n_tok]  # (B, n_tok, embed_dim)
            roi_recon = decoder(roi_tokens)  # (B, n_voxels_roi)
            roi_recons.append(roi_recon)
            token_idx += n_tok

        # 5. Assemble back to flat fMRI vector (inline to preserve dtype under AMP)
        x_recon = torch.zeros(
            B, self.n_voxels, device=roi_recons[0].device,
            dtype=roi_recons[0].dtype)
        for roi, vec in zip(self.decomposer.rois, roi_recons):
            x_recon[:, roi.indices] = vec
        return x_recon

    def forward(self, x: torch.Tensor):
        """Full forward pass."""
        x_recon, z, mu, logvar = None, None, None, None
        z, mu, logvar = self.encode(x, sample_posterior=True)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.01,
        lambda_pcc: float = 0.5,
    ) -> dict:
        """Compute VAE loss: MSE + β·KL + λ_pcc·(1-PCC).

        Optionally includes per-ROI PCC for monitoring.
        """
        # MSE reconstruction
        mse = F.mse_loss(x_recon, x)

        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # PCC loss (sample-wise)
        x_zm = x - x.mean(dim=1, keepdim=True)
        r_zm = x_recon - x_recon.mean(dim=1, keepdim=True)
        pcc = F.cosine_similarity(x_zm, r_zm, dim=1).mean()
        pcc_loss = 1.0 - pcc

        # Total loss
        loss = mse + beta * kl + lambda_pcc * pcc_loss

        return {
            "loss": loss,
            "mse": mse,
            "kl": kl,
            "pcc_loss": pcc_loss,
            "pcc": pcc,
        }

    def param_count(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Per-component breakdown
        enc_mlp = sum(p.numel() for enc in self.roi_encoders
                      for p in enc.parameters())
        dec_mlp = sum(p.numel() for dec in self.roi_decoders
                      for p in dec.parameters())
        transformer = sum(
            p.numel() for blk in self.encoder_blocks for p in blk.parameters()
        ) + sum(
            p.numel() for blk in self.decoder_blocks for p in blk.parameters()
        )

        return {
            "total": total,
            "trainable": trainable,
            "total_mb": total * 4 / 1024 / 1024,
            "roi_encoders": enc_mlp,
            "roi_decoders": dec_mlp,
            "transformer": transformer,
        }


# ─── Factory ─────────────────────────────────────────────────────────────────


def create_fmri_roi_vae(**kwargs) -> FmriROIVAE:
    """Create FmriROIVAE from keyword arguments."""
    return FmriROIVAE(FmriROIVAEConfig(**kwargs))
