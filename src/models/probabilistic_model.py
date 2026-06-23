"""
Probabilistic iTransformer + NHiTS with Multi-Quantile Output Head.

Replaces the deterministic NHiTSHead projection with a QuantileProjectionHead
that outputs 4 quantiles (P10, P50, P90, P99) per horizon.

Loss function: Pinball Loss (Quantile Loss).
Monotonicity enforced via cumulative softplus on the quantile axis.

Quantiles: [0.10, 0.50, 0.90, 0.99]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import iTransformer backbone from the companion repo
_ITRANSFORMER_REPO = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "iTransformer"
)
if _ITRANSFORMER_REPO not in sys.path:
    sys.path.insert(0, _ITRANSFORMER_REPO)

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

QUANTILES = [0.10, 0.50, 0.90, 0.99]
QUANTILE_LABELS = ["P10", "P50", "P90", "P99"]

# GEFCom2014 full quantile set (0.01 to 0.99)
GEFCOM_QUANTILES = [round(i * 0.01, 2) for i in range(1, 100)]

# ═════════════════════════════════════════════════════════════════════════════
# QuantileProjectionHead
# ═════════════════════════════════════════════════════════════════════════════

class QuantileProjectionHead(nn.Module):
    """
    Replaces the standard linear output head.

    Input:  hidden_dim features from the model backbone
    Output: (horizon × n_quantiles) values reshaped to (B, horizon, n_quantiles)
    """
    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        quantiles: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = len(quantiles or QUANTILES)
        self.proj = nn.Linear(hidden_dim, horizon * self.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        return out.view(x.shape[0], self.horizon, self.n_quantiles)


# ═════════════════════════════════════════════════════════════════════════════
# Pinball Loss
# ═════════════════════════════════════════════════════════════════════════════

class PinballLoss(nn.Module):
    """
    Multi-quantile pinball loss.

    Computes the average pinball loss across all specified quantiles.

    For a single quantile q ∈ (0,1):
        L_q(y, ŷ) = q × max(y - ŷ, 0) + (1-q) × max(ŷ - y, 0)
    """
    def __init__(self, quantiles: list[float] | None = None) -> None:
        super().__init__()
        self.quantiles = torch.tensor(quantiles or QUANTILES, dtype=torch.float32)

    def forward(
        self,
        predictions: torch.Tensor,  # (batch, horizon, n_quantiles)
        targets: torch.Tensor,       # (batch, horizon)
    ) -> torch.Tensor:
        if predictions.shape[-1] != len(self.quantiles):
            raise ValueError(
                f"predictions last dim {predictions.shape[-1]} != "
                f"n_quantiles {len(self.quantiles)}"
            )
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)
        q = self.quantiles.to(predictions.device)

        errors = targets_expanded - predictions
        loss = torch.max(q * errors, (q - 1) * errors)
        return loss.mean()


# ═════════════════════════════════════════════════════════════════════════════
# Monotonicity Enforcement
# ═════════════════════════════════════════════════════════════════════════════

def enforce_quantile_monotonicity(
    preds: torch.Tensor,  # (batch, horizon, n_quantiles) — expected order: P10, P50, P90, P99
) -> torch.Tensor:
    """
    Ensures quantile predictions are non-decreasing across the quantile axis.

    Uses cumulative softplus to guarantee strict ordering:
        P10 ≤ P50 ≤ P90 ≤ P99
    """
    base = preds[..., :1]
    deltas = F.softplus(preds[..., 1:] - preds[..., :-1])
    increments = torch.cat([base, deltas], dim=-1)
    return torch.cumsum(increments, dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
# iTransformer + NHiTS Probabilistic
# ═════════════════════════════════════════════════════════════════════════════

class iTransformerNHiTS_Probabilistic(nn.Module):
    """
    iTransformer backbone with a multi-quantile output head.

    Architecture:
        1. DataEmbedding_inverted: (B, L, N) → (B, N, D)
        2. Encoder: (B, N, D) → (B, N, D)
        3. Variate pooling: (B, N, D) → (B, D)
        4. QuantileProjectionHead: (B, D) → (B, S, n_quantiles)
        5. Monotonicity enforcement

    Args (via config object):
        seq_len:         Input look-back window length
        pred_len:        Forecast horizon
        enc_in:          Number of input variates
        d_model:         Embedding / encoder dimension
        n_heads:         Number of attention heads
        e_layers:        Number of encoder layers
        d_ff:            Feed-forward dimension
        dropout:         Dropout rate
        activation:      Activation function ('gelu' or 'relu')
        embed:           Time feature encoding type ('timeF' or 'fixed')
        freq:            Frequency for time features
        class_strategy:  Pooling strategy ('projection', 'average', 'cls_token')
        use_norm:        Whether to use non-stationary normalization
    """
    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.enc_in = configs.enc_in

        # Quantile config
        self.quantiles = getattr(configs, 'quantiles', QUANTILES)
        self.n_quantiles = len(self.quantiles)
        self.target_channels = getattr(configs, 'target_channels', None)  # indices of target variates
        self.n_targets = getattr(configs, 'n_targets', 1) if self.target_channels is None else len(self.target_channels)
        self.ws_channel = getattr(configs, 'ws_channel', -1)

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Per-variate projection: (B, N, D) → (B, N)
        self.variate_proj = nn.Linear(configs.d_model, 1)

        # Quantile head: produces (B, S, nq) for single target or (B, N_targets, S, nq)
        if self.n_targets == 1:
            self.quantile_head = QuantileProjectionHead(
                hidden_dim=configs.enc_in,
                horizon=configs.pred_len,
                quantiles=self.quantiles,
            )
        else:
            # Per-target quantile projection
            # Input: (B, N_targets), Output: (B, N_targets, S, nq)
            self.target_quantile_proj = nn.Linear(1, configs.pred_len * self.n_quantiles)
            self.quantile_head = None

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, N = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        h = self.variate_proj(enc_out[:, :N, :]).squeeze(-1)  # (B, N)

        if self.n_targets > 1 and self.target_channels is not None:
            h = h[:, self.target_channels]  # (B, N_targets)
            # Per-target quantile projection
            out = self.target_quantile_proj(h.unsqueeze(-1))  # (B, N_targets, S*nq)
            B2, NT, _ = out.shape
            quantiles = out.view(B2, NT, self.pred_len, self.n_quantiles)  # (B, N_targets, S, nq)
        else:
            quantiles = self.quantile_head(h)  # (B, S, nq)

        quantiles = enforce_quantile_monotonicity(quantiles)

        if self.use_norm:
            if self.n_targets > 1 and self.target_channels is not None:
                for i, ci in enumerate(self.target_channels):
                    s = stdev[:, 0, ci:ci+1]  # (B, 1)
                    m = means[:, 0, ci:ci+1]
                    quantiles[:, i] = quantiles[:, i] * s.unsqueeze(-1) + m.unsqueeze(-1)
            else:
                idx = self.ws_channel if self.ws_channel >= 0 else N - 1
                s = stdev[:, 0, idx:idx+1]
                m = means[:, 0, idx:idx+1]
                quantiles = quantiles * s.unsqueeze(1) + m.unsqueeze(1)

        return quantiles

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask=None,
    ) -> torch.Tensor:
        return self.forecast(x_enc, x_mark_enc)
