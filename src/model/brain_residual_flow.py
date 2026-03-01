"""
BrainResidualFlow — Residual Flow Matching Architecture.

End-to-End model that splits fMRI latent prediction into two parts:
1. Regression (z_bar): Deterministic prediction using a Transformer Encoder.
2. Residual Flow: Matches the residual (z_true - z_bar) using OT Flow Matching.

Both branches share the same DINOv2 Layer Mixing and Perceiver Bottleneck (Context Encoder).
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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



@dataclass
class BrainResidualFlowConfig:
    # Latent configuration
    latent_dim: int = 768
    n_latent_tokens: int = 12       # 768 = 12 * 64

    # Context (DINOv2) configuration
    context_dim: int = 768
    n_dino_layers: int = 4
    n_context_queries: int = 257    # How many tokens to pass to branches (up to 257)

    # Context Bottleneck (Shared)
    # If we want to use Perceiver, we can add it. For now, we just project.
    
    # Regression Branch (Transformer Encoder)
    reg_hidden_dim: int = 512
    reg_depth: int = 4
    reg_num_heads: int = 8
    
    # Flow Branch (DiT)
    flow_hidden_dim: int = 512
    flow_depth: int = 4
    flow_num_heads: int = 8
    
    # Shared
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1


class ContextEncoder(nn.Module):
    """Mixes mult-layer DINOv2 and optionally compresses via Perceiver."""
    def __init__(self, config: BrainResidualFlowConfig):
        super().__init__()
        self.config = config
        self.layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))
        
        # If we just keep 257 tokens, we only need a projection to match hidden dims
        # Actually, since Reg and Flow might have different hidden_dims, we project later
        
    def forward(self, dino_multilayer):
        """Returns (B, 257, context_dim)"""
        w = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        return (dino_multilayer * w).sum(dim=1)


class RegressionEncoder(nn.Module):
    """Transformer Encoder predicting z_bar from context tokens."""
    def __init__(self, config: BrainResidualFlowConfig):
        super().__init__()
        D = config.reg_hidden_dim
        
        self.input_proj = nn.Linear(config.context_dim, D)
        self.pos_embed = nn.Parameter(torch.randn(1, config.n_context_queries, D) * 0.02)
        
        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=config.reg_num_heads,
            dim_feedforward=int(D * config.mlp_ratio),
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.reg_depth)
        
        self.norm = nn.LayerNorm(D)
        
        # Output head: Global Average Pooling -> Linear -> 768
        self.head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, config.latent_dim)
        )
        # Zero init final layer for safe start
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, context):
        """context: (B, 257, C)"""
        x = self.input_proj(context) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        
        # GAP
        x_pooled = x.mean(dim=1)
        return self.head(x_pooled)


class FlowPrefixDiT(nn.Module):
    """Flow Prefix DiT matching z_res."""
    def __init__(self, config: BrainResidualFlowConfig):
        super().__init__()
        self.config = config
        D = config.flow_hidden_dim
        
        # Context projection
        self.context_proj = nn.Linear(config.context_dim, D)
        self.context_pos_embed = nn.Parameter(torch.randn(1, config.n_context_queries, D) * 0.02)
        self.context_mask_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # Timestep
        self.t_embedder = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D)
        )
        
        # Latent tokens
        assert config.latent_dim % config.n_latent_tokens == 0
        self.token_dim = config.latent_dim // config.n_latent_tokens
        self.latent_proj = nn.Linear(self.token_dim, D)
        self.latent_pos_embed = nn.Parameter(torch.randn(1, config.n_latent_tokens, D) * 0.02)
        
        # Blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.flow_depth)]
        self.blocks = nn.ModuleList([
            BrainPrefixDiTBlock(
                hidden_dim=D, num_heads=config.flow_num_heads,
                mlp_ratio=config.mlp_ratio, dropout=config.dropout, drop_path=dpr[i]
            ) for i in range(config.flow_depth)
        ])
        
        # Output head
        self.final_layer_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.output_proj = nn.Linear(D, self.token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, t, z_t, context, mask_ratio=0.0):
        B = z_t.shape[0]
        N = self.config.n_latent_tokens
        
        # Context
        c = self.context_proj(context) + self.context_pos_embed
        if mask_ratio > 0.0 and self.training:
            mask = torch.rand(B, self.config.n_context_queries, device=z_t.device) < mask_ratio
            mask_tokens = self.context_mask_token.expand(B, self.config.n_context_queries, -1)
            c = torch.where(mask.unsqueeze(-1), mask_tokens, c)
            
        # Timestep
        t_emb = timestep_embedding(t * 1000, self.config.flow_hidden_dim)
        t_cond = self.t_embedder(t_emb)
        
        # Latent
        z_seq = z_t.view(B, N, self.token_dim)
        x = self.latent_proj(z_seq) + self.latent_pos_embed
        
        # Concat & Pass
        seq = torch.cat([c, x], dim=1)
        for block in self.blocks:
            seq = block(seq, t_cond)
            
        # Extract & Output
        x_out = seq[:, -N:, :]
        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_out), shift, scale)
        return self.output_proj(x_out).view(B, -1)

    def forward_with_cfg(self, t, z_t, context, cfg_scale=1.0):
        if cfg_scale == 1.0:
            return self.forward(t, z_t, context)
            
        B = z_t.shape[0]
        N = self.config.n_latent_tokens
        
        # Context
        c_cond = self.context_proj(context) + self.context_pos_embed
        c_uncond = self.context_mask_token.expand(B, self.config.n_context_queries, -1).clone()
        c_batched = torch.cat([c_cond, c_uncond], dim=0)
        
        t_emb = timestep_embedding(t * 1000, self.config.flow_hidden_dim)
        t_cond = self.t_embedder(t_emb)
        t_cond = torch.cat([t_cond, t_cond], dim=0)
        
        z_seq = z_t.view(B, N, self.token_dim)
        x = self.latent_proj(z_seq) + self.latent_pos_embed
        x_batched = torch.cat([x, x.clone()], dim=0)
        
        seq = torch.cat([c_batched, x_batched], dim=1)
        for block in self.blocks:
            seq = block(seq, t_cond)
            
        x_out = seq[:, -N:, :]
        mod_params = self.final_adaLN(t_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x_out = modulate(self.final_layer_norm(x_out), shift, scale)
        v_pred = self.output_proj(x_out).view(2 * B, -1)
        
        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)


class BrainResidualFlow(nn.Module):
    """Wrapper combining Shared Context, Regression, and Flow."""
    def __init__(self, config: BrainResidualFlowConfig):
        super().__init__()
        self.config = config
        self.context_encoder = ContextEncoder(config)
        self.regression_branch = RegressionEncoder(config)
        self.flow_branch = FlowPrefixDiT(config)
        
    def forward_regression(self, dino_multilayer):
        context = self.context_encoder(dino_multilayer)  # (B, 257, C)
        return self.regression_branch(context)
        
    def forward_flow(self, t, z_t, dino_multilayer, mask_ratio=0.0):
        with torch.no_grad():
            context = self.context_encoder(dino_multilayer)
        return self.flow_branch(t, z_t, context, mask_ratio)
        
    def forward_flow_with_cfg(self, t, z_t, dino_multilayer, cfg_scale=1.0):
        context = self.context_encoder(dino_multilayer)
        return self.flow_branch.forward_with_cfg(t, z_t, context, cfg_scale)

    def get_layer_mixing_weights(self):
        with torch.no_grad():
            w = F.softmax(self.context_encoder.layer_weights, dim=0).unsqueeze(0).cpu()
        return {'shared': w}

    def param_count(self):
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        reg_p = sum(p.numel() for p in self.regression_branch.parameters() if p.requires_grad)
        flow_p = sum(p.numel() for p in self.flow_branch.parameters() if p.requires_grad)
        ctx_p = sum(p.numel() for p in self.context_encoder.parameters() if p.requires_grad)
        return {
            'ctx_M': ctx_p / 1e6,
            'reg_M': reg_p / 1e6,
            'flow_M': flow_p / 1e6,
            'total_M': total_p / 1e6
        }
