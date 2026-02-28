"""
BrainMaskedFlowDiT — Flow Matching DiT with Token Masking Regularization.

Combines the flow matching framework from BrainFlowDiT with token masking:
  - Training: randomly mask a fraction of latent tokens, forcing the model
    to rely on DINOv2 context rather than copying input.
  - Inference: no masking, standard ODE integration.

Architecture (from BrainFlowDiT):
  1. DiT backbone with AdaLN-Zero
  2. Dual conditioning: Global (t + CLS) + Dense (cross-attention to patches)
  3. Informed prior regressor
  4. Learnable DINOv2 layer mixing

New additions:
  - Learnable mask token
  - Configurable mask_ratio (training only)
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

    # Informed Prior Regressor
    use_regressor: bool = True
    regressor_depth: int = 2

    # Masking (NEW)
    mask_ratio: float = 0.25        # Fraction of latent tokens to mask during training


# ─── DiT Block ────────────────────────────────────────────────────────────────


class BrainDiTBlock(nn.Module):
    """DiT block with Self-Attention, Cross-Attention, and AdaLN-Zero."""
    def __init__(self, hidden_dim, context_dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path

        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-Attention to Dense Context
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )

        # AdaLN-Zero params (for norm1, norm2, norm3)
        # 6 params for shift/scale, + 3 for gate values (alpha)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim + 3)
        )
        # Initialize AdaLN-Zero
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, global_cond, dense_cond):
        """
        x: (B, N, D)
        global_cond: (B, D) from t_emb + CLS
        dense_cond: (B, N_ctx, C) mapped to D
        """
        mod_params = self.adaLN_modulation(global_cond)

        # Split params: 6 scale/shift, 3 gates
        splits = mod_params.split(self.hidden_dim, dim=-1)
        shift1, scale1, shift2, scale2, shift3, scale3 = splits[:6]
        gate1, gate2, gate3 = mod_params[..., -3:].split(1, dim=-1)

        gate1 = gate1.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)
        gate3 = gate3.unsqueeze(1)

        # 1. Self-Attention
        h1 = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + drop_path(attn_out * gate1, self.drop_path_rate, self.training)

        # 2. Cross-Attention
        h2 = modulate(self.norm2(x), shift2, scale2)
        cross_out, _ = self.cross_attn(h2, dense_cond, dense_cond)
        x = x + drop_path(cross_out * gate2, self.drop_path_rate, self.training)

        # 3. FFN
        h3 = modulate(self.norm3(x), shift3, scale3)
        mlp_out = self.mlp(h3)
        x = x + drop_path(mlp_out * gate3, self.drop_path_rate, self.training)

        return x


# ─── Main Model ───────────────────────────────────────────────────────────────


class BrainMaskedFlowDiT(nn.Module):
    """
    Brain Masked Flow Diffusion Transformer.
    Flow matching with token masking regularization.
    """
    def __init__(self, config: BrainMaskedFlowDiTConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.context_dim

        # ─── 1. DINOv2 Layer Mixing ───
        self.reg_layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))
        self.flow_layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))

        # ─── 2. Conditional Encoders ───
        # Global Conditioning: t_emb + CLS
        self.t_embedder = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D)
        )
        self.cls_embedder = nn.Sequential(
            nn.Linear(C, D),
            nn.GELU(approximate='tanh'),
            nn.Linear(D, D)
        )
        # Dense Conditioning: Spatial Patches
        self.patch_embedder = nn.Sequential(
            nn.Linear(C, D),
            nn.GELU(approximate='tanh'),
            nn.Linear(D, D)
        )

        # ─── 3. Latent Tokenization ───
        assert config.latent_dim % config.n_latent_tokens == 0
        self.token_dim = config.latent_dim // config.n_latent_tokens
        self.latent_proj = nn.Linear(self.token_dim, D)
        self.pos_embed = nn.Parameter(torch.randn(1, config.n_latent_tokens, D) * 0.02)

        # ─── 4. Mask Token (NEW) ───
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.mask_token, std=0.02)

        # ─── 5. Backbone DiT Blocks ───
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            BrainDiTBlock(
                hidden_dim=D,
                context_dim=D,
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

        # ─── 7. Informed Regressor (Optional) ───
        if config.use_regressor:
            reg_layers = []
            in_dim = C * 2
            for _ in range(config.regressor_depth):
                reg_layers.extend([
                    nn.Linear(in_dim, D),
                    nn.GELU(approximate='tanh'),
                    nn.Dropout(config.dropout)
                ])
                in_dim = D
            reg_layers.append(nn.Linear(D, config.latent_dim))
            self.regressor = nn.Sequential(*reg_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        if self.config.use_regressor:
            nn.init.zeros_(self.regressor[-1].weight)
            nn.init.zeros_(self.regressor[-1].bias)

    def _process_context(self, dino_multilayer, layer_weights):
        """
        Process the multi-layer DINOv2 features using specific layer weights.
        dino_multilayer: (B, L, 257, C)
        Returns:
            cls_token: (B, C), spatial_patches: (B, 256, C)
        """
        w = F.softmax(layer_weights, dim=0).view(1, -1, 1, 1)
        dino_mixed = (dino_multilayer * w).sum(dim=1)
        cls_token = dino_mixed[:, 0, :]
        spatial_patches = dino_mixed[:, 1:, :]
        return cls_token, spatial_patches

    def forward_regression(self, dino_multilayer):
        """Predict the conditional mean z_bar."""
        assert self.config.use_regressor, "Regressor not enabled in config."
        cls_token, spatial_patches = self._process_context(
            dino_multilayer, self.reg_layer_weights)
        pooled_spatial = spatial_patches.mean(dim=1)
        reg_input = torch.cat([cls_token, pooled_spatial], dim=1)
        return self.regressor(reg_input)

    def _apply_mask(self, x, mask_ratio):
        """
        Apply token masking to the projected latent sequence.
        x: (B, N, D) — projected latent tokens (before pos_embed)
        mask_ratio: float — fraction of tokens to mask

        Returns:
            x_masked: (B, N, D) — with masked tokens replaced
            mask: (B, N) — bool tensor, True = masked
        """
        B, N, D = x.shape

        if not self.training or mask_ratio <= 0:
            mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            return x, mask

        # Random mask per sample
        noise = torch.rand(B, N, device=x.device)
        mask = noise < mask_ratio  # (B, N), True = masked

        # Replace masked tokens with learnable mask_token
        mask_expanded = mask.unsqueeze(-1).expand_as(x)  # (B, N, D)
        x_masked = torch.where(mask_expanded, self.mask_token.expand_as(x), x)

        return x_masked, mask

    def forward_flow(self, t, z_t, dino_multilayer, mask_ratio=None):
        """
        Predict the velocity v(z_t, t | DINOv2) with optional token masking.

        Args:
            t: (B,)
            z_t: (B, latent_dim)
            dino_multilayer: (B, L, 257, context_dim)
            mask_ratio: float or None. If None, uses config default during
                        training and 0.0 during eval.

        Returns:
            v_pred: (B, latent_dim) — predicted velocity
            mask: (B, n_latent_tokens) — which tokens were masked
        """
        B = z_t.shape[0]

        # Determine mask ratio
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio if self.training else 0.0

        # 1. Context embedding
        cls_token, spatial_patches = self._process_context(
            dino_multilayer, self.flow_layer_weights)
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        global_cond = self.t_embedder(t_emb) + self.cls_embedder(cls_token)
        dense_cond = self.patch_embedder(spatial_patches)

        # 2. Tokenize Latent
        N = self.config.n_latent_tokens
        T_dim = self.token_dim
        z_seq = z_t.view(B, N, T_dim)
        x = self.latent_proj(z_seq)  # (B, N, D)

        # 3. Apply masking (NEW — before pos_embed)
        x, mask = self._apply_mask(x, mask_ratio)

        # 4. Add positional embedding
        x = x + self.pos_embed

        # 5. Backbone
        for block in self.blocks:
            x = block(x, global_cond, dense_cond)

        # 6. Output Head
        mod_params = self.final_adaLN(global_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x = modulate(self.final_layer_norm(x), shift, scale)
        x = self.output_proj(x)  # (B, N, token_dim)

        # Flatten back to (B, latent_dim)
        return x.view(B, -1), mask

    def forward_flow_with_cfg(self, t, z_t, dino_multilayer, cfg_scale=1.0):
        """CFG-guided velocity prediction (no masking at inference)."""
        if cfg_scale == 1.0:
            v, _ = self.forward_flow(t, z_t, dino_multilayer, mask_ratio=0.0)
            return v
        v_cond, _ = self.forward_flow(t, z_t, dino_multilayer, mask_ratio=0.0)
        v_uncond, _ = self.forward_flow(t, z_t, torch.zeros_like(dino_multilayer),
                                        mask_ratio=0.0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def get_layer_mixing_weights(self):
        """Returns the layer weights dict for consistent logging format."""
        with torch.no_grad():
            w_reg = F.softmax(self.reg_layer_weights, dim=0).unsqueeze(0).cpu()
            w_flow = F.softmax(self.flow_layer_weights, dim=0).unsqueeze(0).cpu()
        return {
            'reg': w_reg.repeat(self.config.regressor_depth, 1),
            'flow': w_flow.repeat(self.config.depth, 1)
        }

    def param_count(self):
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        reg_p = (sum(p.numel() for p in self.regressor.parameters() if p.requires_grad)
                 if self.config.use_regressor else 0)
        return {
            'reg_M': reg_p / 1e6,
            'flow_M': (total_p - reg_p) / 1e6,
            'shared_M': 0.0,
            'total_M': total_p / 1e6
        }
