"""
BrainXAttnFlowDiT — Cross-Attention Flow Matching DiT with Self-Conditioning.

Conditional flow matching: N(0,I) → z_true (fMRI latent),
conditioned on DINOv2 multi-layer features via Cross-Attention + Perceiver Bottleneck.

Key differences from BrainOTFlowDiT:
  1. Cross-Attention (not prefix concatenation) — fMRI tokens attend to DINOv2 separately
  2. Perceiver Bottleneck — compress 257 DINOv2 tokens to K=32 learnable queries
  3. Self-Conditioning — model can receive its own previous estimate as input
  4. No reconstruction loss dependency — pure velocity matching

Architecture per block:
  Self-Attn(fMRI 12 tokens) → Cross-Attn(fMRI↔DINOv2 32 tokens) → FFN
  All with AdaLN-Zero (timestep-conditioned)
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Utilities ────────────────────────────────────────────────────────────────


def drop_path(x, drop_prob: float = 0., training: bool = False,
              scale_by_keep: bool = True):
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
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BrainXAttnFlowDiTConfig:
    # Latent configuration
    latent_dim: int = 768
    n_latent_tokens: int = 12       # 768 = 12 * 64

    # Context (DINOv2) configuration
    context_dim: int = 768
    n_dino_layers: int = 4
    n_context_queries: int = 32     # Perceiver bottleneck output tokens

    # Transformer backbone
    hidden_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1

    # Self-conditioning
    use_self_cond: bool = True


# ─── Perceiver Bottleneck ─────────────────────────────────────────────────────


class ContextBottleneck(nn.Module):
    """Perceiver Resampler: compress 257 DINOv2 tokens to K learnable queries.

    Implements an information bottleneck on the visual context, preventing
    the flow model from memorizing individual training samples by forcing
    context through a compressed representation.
    """

    def __init__(self, n_queries: int, context_dim: int, hidden_dim: int,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_queries = n_queries

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)

        # Project raw DINOv2 (context_dim) to hidden_dim
        self.kv_proj = nn.Linear(context_dim, hidden_dim)

        # Stack of cross-attention layers for multi-round refinement
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm_q': nn.LayerNorm(hidden_dim),
                'norm_kv': nn.LayerNorm(hidden_dim),
                'cross_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'norm_ff': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(approximate='tanh'),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(self, dino_tokens):
        """
        Args:
            dino_tokens: (B, 257, context_dim) — mixed DINOv2 features
        Returns:
            compressed: (B, K, hidden_dim) — K compressed context tokens
        """
        B = dino_tokens.shape[0]
        kv = self.kv_proj(dino_tokens)  # (B, 257, D)
        q = self.queries.expand(B, -1, -1)  # (B, K, D)

        for layer in self.layers:
            # Cross-attention: queries attend to DINOv2 tokens
            q_norm = layer['norm_q'](q)
            kv_norm = layer['norm_kv'](kv)
            attn_out, _ = layer['cross_attn'](q_norm, kv_norm, kv_norm)
            q = q + attn_out

            # FFN
            q = q + layer['ffn'](layer['norm_ff'](q))

        return q  # (B, K, D)


# ─── Cross-Attention DiT Block ────────────────────────────────────────────────


class CrossAttnDiTBlock(nn.Module):
    """DiT block with separated Self-Attention and Cross-Attention.

    Architecture:
      1. Self-Attention over fMRI latent tokens only (12 tokens)
      2. Cross-Attention: fMRI queries → compressed DINOv2 key/values (32 tokens)
      3. FFN

    All three sub-layers use AdaLN-Zero modulation (timestep-conditioned).
    """

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0,
                 dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path_rate

        # 1. Self-Attention (fMRI tokens only)
        self.norm_sa = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # 2. Cross-Attention (fMRI → DINOv2)
        self.norm_ca_q = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_ca_kv = nn.LayerNorm(hidden_dim)  # affine for context
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # 3. FFN
        self.norm_ff = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero: 6 shift/scale params + 3 gate scalars
        # [shift_sa, scale_sa, shift_ca, scale_ca, shift_ff, scale_ff, gate_sa, gate_ca, gate_ff]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim + 3),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, context, t_cond):
        """
        x: (B, N, D) — fMRI latent tokens (N=12)
        context: (B, K, D) — compressed DINOv2 tokens (K=32)
        t_cond: (B, D) — timestep embedding
        """
        # Parse AdaLN modulation parameters
        mod = self.adaLN_modulation(t_cond)
        D = self.hidden_dim
        shift_sa = mod[:, 0*D:1*D]
        scale_sa = mod[:, 1*D:2*D]
        shift_ca = mod[:, 2*D:3*D]
        scale_ca = mod[:, 3*D:4*D]
        shift_ff = mod[:, 4*D:5*D]
        scale_ff = mod[:, 5*D:6*D]
        gates = mod[:, 6*D:]  # (B, 3)
        gate_sa = gates[:, 0:1].unsqueeze(1)   # (B, 1, 1)
        gate_ca = gates[:, 1:2].unsqueeze(1)
        gate_ff = gates[:, 2:3].unsqueeze(1)

        # 1. Self-Attention (fMRI tokens only, seq_len=12)
        h = modulate(self.norm_sa(x), shift_sa, scale_sa)
        sa_out, _ = self.self_attn(h, h, h)
        x = x + drop_path(sa_out * gate_sa, self.drop_path_rate, self.training)

        # 2. Cross-Attention (fMRI queries → DINOv2 keys/values)
        h = modulate(self.norm_ca_q(x), shift_ca, scale_ca)
        ctx = self.norm_ca_kv(context)
        ca_out, _ = self.cross_attn(h, ctx, ctx)
        x = x + drop_path(ca_out * gate_ca, self.drop_path_rate, self.training)

        # 3. FFN
        h = modulate(self.norm_ff(x), shift_ff, scale_ff)
        ff_out = self.ffn(h)
        x = x + drop_path(ff_out * gate_ff, self.drop_path_rate, self.training)

        return x


# ─── Main Model ───────────────────────────────────────────────────────────────


class BrainXAttnFlowDiT(nn.Module):
    """Cross-Attention Flow Matching DiT with Self-Conditioning.

    Flow: N(0,I) → z_true, conditioned on DINOv2 multi-layer features.
    """

    def __init__(self, config: BrainXAttnFlowDiTConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.context_dim

        # ─── 1. DINOv2 Layer Mixing ───
        self.layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))

        # ─── 2. Context Bottleneck (Perceiver) ───
        self.context_bottleneck = ContextBottleneck(
            n_queries=config.n_context_queries,
            context_dim=C,
            hidden_dim=D,
            num_heads=config.num_heads,
            num_layers=2,
            dropout=config.dropout,
        )

        # ─── 3. Timestep Embedder ───
        self.t_embedder = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        # ─── 4. Latent Tokenization ───
        assert config.latent_dim % config.n_latent_tokens == 0
        self.token_dim = config.latent_dim // config.n_latent_tokens
        self.latent_proj = nn.Linear(self.token_dim, D)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, config.n_latent_tokens, D) * 0.02)

        # ─── 5. Self-Conditioning Projector ───
        if config.use_self_cond:
            self.self_cond_proj = nn.Linear(self.token_dim, D)
            nn.init.zeros_(self.self_cond_proj.weight)
            nn.init.zeros_(self.self_cond_proj.bias)

        # ─── 6. Null context for CFG (unconditional) ───
        self.null_context = nn.Parameter(
            torch.randn(1, config.n_context_queries, D) * 0.02)

        # ─── 7. Backbone DiT Blocks ───
        dpr = [x.item() for x in torch.linspace(
            0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(
                hidden_dim=D,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path_rate=dpr[i],
            ) for i in range(config.depth)
        ])

        # ─── 8. Final Output Head ───
        self.final_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D, 2 * D),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.output_proj = nn.Linear(D, self.token_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _process_context(self, dino_multilayer):
        """Mix multi-layer DINOv2 features → (B, 257, C)."""
        w = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        return (dino_multilayer * w).sum(dim=1)

    def _get_context(self, dino_multilayer):
        """Full context pipeline: layer mix → bottleneck → (B, K, D)."""
        dino_mixed = self._process_context(dino_multilayer)  # (B, 257, C)
        return self.context_bottleneck(dino_mixed)  # (B, K, D)

    def forward_flow(self, t, z_t, dino_multilayer, z_self_cond=None):
        """
        Predict velocity v(z_t, t | DINOv2).

        Args:
            t: (B,) timestep
            z_t: (B, latent_dim) noisy state
            dino_multilayer: (B, L, 257, context_dim) multi-layer DINOv2
            z_self_cond: (B, latent_dim) optional self-conditioning estimate,
                         or None / zeros for no self-conditioning

        Returns:
            v_pred: (B, latent_dim) predicted velocity
        """
        B = z_t.shape[0]
        N = self.config.n_latent_tokens

        # 1. Context through bottleneck
        context = self._get_context(dino_multilayer)  # (B, K, D)

        # 2. Timestep conditioning
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        t_cond = self.t_embedder(t_emb)  # (B, D)

        # 3. Tokenize latent
        z_seq = z_t.view(B, N, self.token_dim)
        x = self.latent_proj(z_seq) + self.latent_pos_embed  # (B, N, D)

        # 4. Self-conditioning: add projected self-estimate
        if self.config.use_self_cond and z_self_cond is not None:
            sc_seq = z_self_cond.view(B, N, self.token_dim)
            x = x + self.self_cond_proj(sc_seq)

        # 5. Backbone (Cross-Attention DiT)
        for block in self.blocks:
            x = block(x, context, t_cond)

        # 6. Output head with AdaLN modulation
        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_norm(x), shift, scale)
        x_out = self.output_proj(x_out)  # (B, N, token_dim)

        return x_out.view(B, -1)

    def forward_flow_with_cfg(self, t, z_t, dino_multilayer,
                              cfg_scale=1.0, z_self_cond=None):
        """CFG-guided velocity prediction.

        Runs conditional and unconditional forward passes in a single batch,
        then interpolates: v = v_uncond + cfg_scale * (v_cond - v_uncond)
        """
        if cfg_scale == 1.0:
            return self.forward_flow(t, z_t, dino_multilayer,
                                     z_self_cond=z_self_cond)

        B = z_t.shape[0]
        N = self.config.n_latent_tokens

        # Context: conditional + unconditional (null)
        context_cond = self._get_context(dino_multilayer)  # (B, K, D)
        context_uncond = self.null_context.expand(B, -1, -1)
        context_batched = torch.cat([context_cond, context_uncond], dim=0)

        # Timestep (doubled)
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        t_cond = self.t_embedder(t_emb)
        t_cond = torch.cat([t_cond, t_cond], dim=0)

        # Latent (doubled)
        z_seq = z_t.view(B, N, self.token_dim)
        x = self.latent_proj(z_seq) + self.latent_pos_embed

        # Self-conditioning (applied to both branches)
        if self.config.use_self_cond and z_self_cond is not None:
            sc_seq = z_self_cond.view(B, N, self.token_dim)
            x = x + self.self_cond_proj(sc_seq)

        x = torch.cat([x, x.clone()], dim=0)  # (2B, N, D)

        # Backbone
        for block in self.blocks:
            x = block(x, context_batched, t_cond)

        # Output head
        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_norm(x), shift, scale)
        v_pred = self.output_proj(x_out).view(2 * B, -1)

        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def get_layer_mixing_weights(self):
        """Returns layer mixing weights for logging."""
        with torch.no_grad():
            w = F.softmax(self.layer_weights, dim=0).unsqueeze(0).cpu()
        return {'flow': w.repeat(self.config.depth, 1)}

    def param_count(self):
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bottleneck_p = sum(
            p.numel() for p in self.context_bottleneck.parameters()
            if p.requires_grad)
        backbone_p = total_p - bottleneck_p
        return {
            'flow_M': backbone_p / 1e6,
            'bottleneck_M': bottleneck_p / 1e6,
            'total_M': total_p / 1e6,
        }
