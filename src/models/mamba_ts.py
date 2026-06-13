"""src/models/mamba_ts.py — Mamba-based time series forecaster.

Wraps the official mamba-ssm library (requires CUDA, Linux/Windows).
Falls back to a pure-PyTorch S4-like recurrence for CPU testing.

Install on remote desktop:
    pip install mamba-ssm causal-conv1d
"""
import math
import torch
import torch.nn as nn


# ── Try to import the fast CUDA Mamba implementation ─────────────────────────
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ── CPU fallback: simplified selective SSM (no CUDA fusion) ──────────────────
class SimpleMambaBlock(nn.Module):
    """
    Pure-PyTorch Mamba block for CPU smoke-testing.
    NOT optimized — use mamba_ssm on the GPU machine.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_model * expand
        self.d_state  = d_state

        self.in_proj   = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d    = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                   padding=d_conv - 1, groups=self.d_inner)
        self.act       = nn.SiLU()
        self.x_proj    = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj   = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj  = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm      = nn.LayerNorm(d_model)

        # Fixed A matrix (diagonal, real)
        A = -torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.register_buffer("A_log", torch.log(-A))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)                         # (B, L, 2*D_inner)
        x_, z = xz.chunk(2, dim=-1)                  # each (B, L, D_inner)

        # Conv along time
        x_ = self.conv1d(x_.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x_ = self.act(x_)

        # Simplified linear recurrence (not selective — placeholder)
        y = x_ * torch.sigmoid(z)
        return self.out_proj(y) + residual


# ─────────────────────────────────────────────────────────────────────────────
# Patch embedding
# ─────────────────────────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Divide L-length time series into non-overlapping patches of size P.
    Embed each patch with a linear layer.
    """

    def __init__(self, patch_size: int, d_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj       = nn.Linear(patch_size * d_in, d_model)
        self.drop       = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  →  (B, num_patches, d_model)
        Pads if L % patch_size != 0.
        """
        B, L, C = x.shape
        pad = (self.patch_size - L % self.patch_size) % self.patch_size
        if pad:
            x = torch.cat([x, x[:, -pad:, :]], dim=1)
        x = x.unfold(1, self.patch_size, self.patch_size)    # (B, N_patches, C, P)
        B, N, C, P = x.shape
        x = x.reshape(B, N, C * P)                           # (B, N, C*P)
        return self.drop(self.norm(self.proj(x)))             # (B, N, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Mamba Time Series Forecaster
# ─────────────────────────────────────────────────────────────────────────────
class MambaForecaster(nn.Module):
    """
    Mamba-based multi-step wind speed forecaster.

    Architecture:
      1. Patch embedding: (B, L, C) → (B, N_patches, d_model)
      2. Stack of N Mamba blocks
      3. Flatten last hidden state → Linear projection → (B, T, C)

    Args:
        seq_len    : look-back window (L)
        pred_len   : forecast horizon (T)
        enc_in     : number of input variates (C)
        d_model    : embedding dimension
        d_state    : SSM state dimension (N in paper)
        d_conv     : local conv width in Mamba block
        expand     : inner expansion factor (E=2 in paper)
        n_layers   : number of stacked Mamba blocks
        patch_size : patch size for tokenization
        dropout    : dropout probability
    """

    def __init__(
        self,
        seq_len:    int,
        pred_len:   int,
        enc_in:     int   = 1,
        d_model:    int   = 64,
        d_state:    int   = 16,
        d_conv:     int   = 4,
        expand:     int   = 2,
        n_layers:   int   = 4,
        patch_size: int   = 16,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.enc_in   = enc_in

        # Patch tokenizer
        self.patch_embed = PatchEmbedding(patch_size, enc_in, d_model, dropout)
        n_patches = math.ceil(seq_len / patch_size)

        # Mamba blocks
        if MAMBA_AVAILABLE:
            self.layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
        else:
            # CPU fallback
            self.layers = nn.ModuleList([
                SimpleMambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])

        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model * n_patches, pred_len * enc_in)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)

        Returns:
            (B, T, C)
        """
        # Tokenize
        out = self.patch_embed(x)            # (B, N_patches, d_model)

        # Selective SSM blocks
        for layer in self.layers:
            out = layer(out)

        out = self.norm(out)                 # (B, N_patches, d_model)
        B, N, D = out.shape
        out = self.dropout(out.reshape(B, N * D))   # (B, N*d_model)
        out = self.head(out)                         # (B, T*C)
        return out.reshape(B, self.pred_len, self.enc_in)
