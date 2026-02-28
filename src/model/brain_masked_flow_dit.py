"""
BrainMaskedFlowDiT — Pure Flow Matching DiT with Reconstruction Loss.

Standard conditional flow matching: N(0,I) → z_true (fMRI latent),
conditioned on DINOv2 multi-layer features via Prefix Token Early Fusion.

Architecture:
  1. Prefix Token fusion: DINOv2 (257 tokens) + fMRI latent (12 tokens)
  2. DiT backbone with AdaLN-Zero (time-conditioned)
  3. Learnable DINOv2 layer mixing
  4. No regression branch, no masking
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
class BrainMaskedFlowDiTConfig:
    # Latent configuration
    latent_dim: int = 768
    n_latent_tokens: int = 12       # 768 = 12 * 64

    # Context (DINOv2) configuration
    context_dim: int = 768
    n_dino_layers: int = 4

    # Transformer backbone
    hidden_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1

    # Legacy (unused but kept for config compatibility)
    use_regressor: bool = False
    regressor_depth: int = 2
    mask_ratio: float = 0.0


# ─── DiT Block ────────────────────────────────────────────────────────────────


class BrainPrefixDiTBlock(nn.Module):
    """DiT block with Early Fusion (Self-Attention only) and AdaLN-Zero.
    Tokens = [DINO Context... + fMRI latent...]"""
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path

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

        # AdaLN-Zero: 4 params for shift/scale + 2 gate values
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim + 2)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, t_cond):
        """x: (B, N_seq, D), t_cond: (B, D)"""
        mod_params = self.adaLN_modulation(t_cond)
        splits = mod_params.split(self.hidden_dim, dim=-1)
        shift1, scale1, shift2, scale2 = splits[:4]
        gate1, gate2 = mod_params[..., -2:].split(1, dim=-1)
        gate1 = gate1.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # Self-Attention
        h1 = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + drop_path(attn_out * gate1, self.drop_path_rate, self.training)

        # FFN
        h2 = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(h2)
        x = x + drop_path(mlp_out * gate2, self.drop_path_rate, self.training)

        return x


# ─── Main Model ───────────────────────────────────────────────────────────────


class BrainMaskedFlowDiT(nn.Module):
    """
    Pure Flow Matching DiT with Prefix Token Early Fusion.
    Standard flow: N(0,I) → z_true, no regression, no masking.
    """
    def __init__(self, config: BrainMaskedFlowDiTConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.context_dim

        # ─── 1. DINOv2 Layer Mixing ───
        self.flow_layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))

        # ─── 2. Timestep Embedder ───
        self.t_embedder = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D)
        )

        # ─── 3. Context Embedder (Prefix Tokens) ───
        self.context_proj = nn.Linear(C, D)
        self.context_pos_embed = nn.Parameter(torch.randn(1, 257, D) * 0.02)

        # ─── 4. Latent Tokenization ───
        assert config.latent_dim % config.n_latent_tokens == 0
        self.token_dim = config.latent_dim // config.n_latent_tokens
        self.latent_proj = nn.Linear(self.token_dim, D)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, config.n_latent_tokens, D) * 0.02)

        # ─── 5. Backbone DiT Blocks ───
        dpr = [x.item() for x in torch.linspace(
            0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            BrainPrefixDiTBlock(
                hidden_dim=D,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path=dpr[i]
            ) for i in range(config.depth)
        ])

        # ─── 6. Final Output Head ───
        self.final_layer_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D, 2 * D)
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.output_proj = nn.Linear(D, self.token_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _process_context(self, dino_multilayer):
        """Mix multi-layer DINOv2 features. Returns (B, 257, C)."""
        w = F.softmax(self.flow_layer_weights, dim=0).view(1, -1, 1, 1)
        return (dino_multilayer * w).sum(dim=1)

    def forward_flow(self, t, z_t, dino_multilayer):
        """
        Predict velocity v(z_t, t | DINOv2).

        Args:
            t: (B,) timestep
            z_t: (B, latent_dim) noisy state
            dino_multilayer: (B, L, 257, context_dim)

        Returns:
            v_pred: (B, latent_dim) predicted velocity
        """
        B = z_t.shape[0]
        N = self.config.n_latent_tokens

        # 1. Context embedding (Prefix Tokens)
        dino_mixed = self._process_context(dino_multilayer)
        context = self.context_proj(dino_mixed) + self.context_pos_embed

        # 2. Timestep conditioning
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        t_cond = self.t_embedder(t_emb)

        # 3. Tokenize latent
        z_seq = z_t.view(B, N, self.token_dim)
        x = self.latent_proj(z_seq) + self.latent_pos_embed

        # 4. Concatenate context + latent
        seq = torch.cat([context, x], dim=1)  # (B, 257+N, D)

        # 5. Backbone
        for block in self.blocks:
            seq = block(seq, t_cond)

        # 6. Extract fMRI tokens
        x_out = seq[:, -N:, :]

        # 7. Output head
        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_out), shift, scale)
        x_out = self.output_proj(x_out)  # (B, N, token_dim)

        return x_out.view(B, -1)

    def forward_flow_with_cfg(self, t, z_t, dino_multilayer, cfg_scale=1.0):
        """CFG-guided velocity prediction."""
        if cfg_scale == 1.0:
            return self.forward_flow(t, z_t, dino_multilayer)

        B = z_t.shape[0] // 2
        N = self.config.n_latent_tokens

        # Context
        dino_mixed = self._process_context(dino_multilayer)
        context = self.context_proj(dino_mixed) + self.context_pos_embed

        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        t_cond = self.t_embedder(t_emb)

        # Latent
        z_seq = z_t.view(2 * B, N, self.token_dim)
        x = self.latent_proj(z_seq) + self.latent_pos_embed

        seq = torch.cat([context, x], dim=1)

        for block in self.blocks:
            seq = block(seq, t_cond)

        x_out = seq[:, -N:, :]

        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_out), shift, scale)
        v_pred = self.output_proj(x_out).view(2 * B, -1)

        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def get_layer_mixing_weights(self):
        """Returns layer weights for logging."""
        with torch.no_grad():
            w = F.softmax(self.flow_layer_weights, dim=0).unsqueeze(0).cpu()
        return {'flow': w.repeat(self.config.depth, 1)}

    def param_count(self):
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'flow_M': total_p / 1e6,
            'total_M': total_p / 1e6
        }
