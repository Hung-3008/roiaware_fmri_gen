"""
BrainFlow — Unified Direct Flow Matching for fMRI Generation.

Inspired by FlowFM (Ukita & Okita), this model performs direct conditional
flow matching from N(0,I) noise to raw fMRI voxel space (~15724 dim),
conditioned on multi-layer DINOv2 features.

No VAE is needed — the model patchifies fMRI directly:
    fMRI (15724) → Pad(15872) → Reshape(128 patches × 124) → Linear(D)
    → DiT Blocks with DINOv2 prefix attention & adaLN-Zero
    → Linear(124) → Reshape(15872) → Crop(15724)

Key design elements from FlowFM:
  - Prefix attention: [DINOv2 context | fMRI tokens] → self-attention
  - adaLN-Zero: timestep modulation via adaptive layer norm
  - DGS (Dynamic Guidance Switching): randomly zero-out condition for CFG
  - OT linear path: x_t = (1-t)*x_0 + t*x_1
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Utilities ────────────────────────────────────────────────────────────────


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BrainFlowConfig:
    # fMRI dimensions
    n_voxels: int = 15724
    patch_size: int = 124       # 15724 → pad to 15872 → 128 patches

    # Transformer
    hidden_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1

    # Context (DINOv2)
    context_dim: int = 768
    n_dino_layers: int = 4
    n_context_queries: int = 257   # 1 CLS + 256 patches


# ─── DiT Block ────────────────────────────────────────────────────────────────


class BrainFlowDiTBlock(nn.Module):
    """DiT block with self-attention and adaLN-Zero conditioning.
    Operates on concatenated sequence: [DINOv2 context | fMRI tokens]."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path_rate

        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )

        # adaLN-Zero: 4 modulation params (shift1, scale1, shift2, scale2) + 2 gates
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim + 2)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, t_cond):
        """
        x: (B, N_total, D) — concatenated [context | fmri_tokens]
        t_cond: (B, D) — timestep embedding
        """
        mod_params = self.adaLN_modulation(t_cond)
        splits = mod_params.split(self.hidden_dim, dim=-1)
        shift1, scale1, shift2, scale2 = splits[:4]
        gate1, gate2 = mod_params[..., -2:].split(1, dim=-1)
        gate1 = gate1.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # Self-Attention with adaLN
        h1 = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + drop_path(attn_out * gate1, self.drop_path_rate, self.training)

        # FFN with adaLN
        h2 = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(h2)
        x = x + drop_path(mlp_out * gate2, self.drop_path_rate, self.training)

        return x


# ─── Context Encoder ──────────────────────────────────────────────────────────


class ContextEncoder(nn.Module):
    """Mixes multi-layer DINOv2 features via learnable softmax weights."""

    def __init__(self, n_dino_layers: int):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(n_dino_layers))

    def forward(self, dino_multilayer):
        """dino_multilayer: (B, n_layers, 257, C) → (B, 257, C)"""
        w = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        return (dino_multilayer * w).sum(dim=1)


# ─── Main Model ──────────────────────────────────────────────────────────────


class BrainFlow(nn.Module):
    """
    Unified Direct Flow Matching for fMRI.

    Flow matching from N(0,I) noise directly to fMRI voxel space,
    conditioned on multi-layer DINOv2 features.
    No VAE needed.
    """

    def __init__(self, config: Optional[BrainFlowConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = BrainFlowConfig(**kwargs)
        self.config = config
        D = config.hidden_dim

        # ── fMRI patchification ──
        if config.n_voxels % config.patch_size != 0:
            self.padded_voxels = ((config.n_voxels // config.patch_size) + 1) * config.patch_size
        else:
            self.padded_voxels = config.n_voxels
        self.num_patches = self.padded_voxels // config.patch_size
        self.pad_len = self.padded_voxels - config.n_voxels

        # Patch embedding: patch_size → D
        self.patch_embed = nn.Linear(config.patch_size, D)
        self.patch_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, D) * 0.02)

        # ── Context (DINOv2) ──
        self.context_encoder = ContextEncoder(config.n_dino_layers)
        self.context_proj = nn.Linear(config.context_dim, D)
        self.context_pos_embed = nn.Parameter(torch.randn(1, config.n_context_queries, D) * 0.02)
        self.context_mask_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # ── Timestep embedding ──
        self.t_embedder = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D)
        )

        # ── DiT Blocks ──
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            BrainFlowDiTBlock(
                hidden_dim=D,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(config.depth)
        ])

        # ── Output head ──
        self.final_layer_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.output_proj = nn.Linear(D, config.patch_size)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, trunc normal for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # Re-zero the output proj and adaLN (already done above, but safety)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

    def _patchify(self, x):
        """(B, n_voxels) → (B, num_patches, patch_size)"""
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        return x.view(x.shape[0], self.num_patches, self.config.patch_size)

    def _unpatchify(self, x):
        """(B, num_patches, patch_size) → (B, n_voxels)"""
        x = x.reshape(x.shape[0], -1)  # (B, padded_voxels)
        if self.pad_len > 0:
            x = x[:, :self.config.n_voxels]
        return x

    def forward(self, t, x_t, dino_multilayer, mask_ratio=0.0):
        """
        Predict velocity v(x_t, t | DINOv2).

        Args:
            t: (B,) timestep values in [0, 1]
            x_t: (B, n_voxels) point on the probability path
            dino_multilayer: (B, n_layers, 257, C) multi-layer DINOv2 features
            mask_ratio: float, probability of masking context tokens (regularization)

        Returns:
            v_pred: (B, n_voxels) predicted velocity field
        """
        B = x_t.shape[0]

        # ── Context tokens ──
        context = self.context_encoder(dino_multilayer)    # (B, 257, C)
        c = self.context_proj(context) + self.context_pos_embed  # (B, 257, D)

        if mask_ratio > 0.0 and self.training:
            mask = torch.rand(B, self.config.n_context_queries, device=x_t.device) < mask_ratio
            mask_tokens = self.context_mask_token.expand(B, self.config.n_context_queries, -1)
            c = torch.where(mask.unsqueeze(-1), mask_tokens, c)

        # ── Timestep conditioning ──
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        t_cond = self.t_embedder(t_emb)  # (B, D)

        # ── fMRI tokens ──
        x_patches = self._patchify(x_t)                   # (B, 128, 124)
        x_tokens = self.patch_embed(x_patches) + self.patch_pos_embed  # (B, 128, D)

        # ── Prefix attention: [context | fmri_tokens] ──
        seq = torch.cat([c, x_tokens], dim=1)  # (B, 257+128, D) = (B, 385, D)

        for block in self.blocks:
            seq = block(seq, t_cond)

        # ── Extract fMRI tokens & output ──
        x_out = seq[:, -self.num_patches:, :]  # (B, 128, D)

        # Final adaLN + projection
        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_out), shift, scale)
        x_out = self.output_proj(x_out)  # (B, 128, 124)

        # Unpatchify → (B, n_voxels)
        v_pred = self._unpatchify(x_out)
        return v_pred

    def forward_with_cfg(self, t, x_t, dino_multilayer, cfg_scale=1.0):
        """
        Classifier-free guidance inference.
        Runs conditional + unconditional forward pass and interpolates.
        """
        if cfg_scale == 1.0:
            return self.forward(t, x_t, dino_multilayer)

        B = x_t.shape[0]

        # Context: conditional vs unconditional
        context = self.context_encoder(dino_multilayer)
        c_cond = self.context_proj(context) + self.context_pos_embed
        c_uncond = self.context_mask_token.expand(B, self.config.n_context_queries, -1).clone()
        c_batched = torch.cat([c_cond, c_uncond], dim=0)  # (2B, 257, D)

        # Timestep (doubled)
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        t_cond = self.t_embedder(t_emb)
        t_cond = torch.cat([t_cond, t_cond], dim=0)  # (2B, D)

        # fMRI tokens (doubled)
        x_patches = self._patchify(x_t)
        x_tokens = self.patch_embed(x_patches) + self.patch_pos_embed
        x_batched = torch.cat([x_tokens, x_tokens.clone()], dim=0)  # (2B, 128, D)

        # Prefix attention
        seq = torch.cat([c_batched, x_batched], dim=1)  # (2B, 385, D)
        for block in self.blocks:
            seq = block(seq, t_cond)

        # Extract & project
        x_out = seq[:, -self.num_patches:, :]
        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_out), shift, scale)
        x_out = self.output_proj(x_out)
        v_pred = self._unpatchify(x_out)  # (2B, n_voxels)

        # CFG interpolation
        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def get_layer_mixing_weights(self):
        """Return DINOv2 layer mixing weights for logging."""
        with torch.no_grad():
            w = F.softmax(self.context_encoder.layer_weights, dim=0).unsqueeze(0).cpu()
        return {'shared': w}

    def param_count(self):
        """Return parameter counts for logging."""
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        block_p = sum(p.numel() for p in self.blocks.parameters() if p.requires_grad)
        ctx_p = sum(p.numel() for p in self.context_encoder.parameters() if p.requires_grad)
        ctx_p += sum(p.numel() for p in self.context_proj.parameters() if p.requires_grad)
        embed_p = sum(p.numel() for p in self.patch_embed.parameters() if p.requires_grad)
        out_p = sum(p.numel() for p in self.output_proj.parameters() if p.requires_grad)
        return {
            'ctx_M': ctx_p / 1e6,
            'blocks_M': block_p / 1e6,
            'embed_M': embed_p / 1e6,
            'output_M': out_p / 1e6,
            'total_M': total_p / 1e6,
        }
