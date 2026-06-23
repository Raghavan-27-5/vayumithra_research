"""
GEFCom2014-W Probabilistic Model: iTransformer + NHiTS + Future Weather.

Architecture:
  1. DataEmbedding_inverted: (B, L, N) -> (B, N, D)
  2. Encoder: (B, N, D) -> (B, N, D)
  3. Weather injection: per-zone 4-var -> proj -> add to TARGETVAR tokens
  4. ProbabilisticNHiTSHead: (B, N, D) -> (B, N, 1, 99)  multi-scale quantile head
  5. Target selection + monotonicity: (B, 10, 1, 99)

Uses true NHiTS 3-stack hierarchical head with quantile output modification.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

_ITRANSFORMER_REPO = str(Path(__file__).resolve().parent.parent.parent.parent / "iTransformer")
if _ITRANSFORMER_REPO not in sys.path:
    sys.path.insert(0, _ITRANSFORMER_REPO)

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

GEFCOM_QUANTILES = [round(i * 0.01, 2) for i in range(1, 100)]


class ProbabilisticNHiTSBlock(nn.Module):
    """
    NHiTSBlock modified to output (B, N, S, nq) quantiles instead of (B, N, S).
    theta_size = max(pred_len // theta_ratio, 1)
    forecast_fc outputs theta_size * n_quantiles, then reshaped.
    """
    def __init__(self, d_model, pred_len, theta_ratio, dropout, n_quantiles=99):
        super().__init__()
        self.pred_len = pred_len
        self.n_quantiles = n_quantiles
        self.theta_size = max(pred_len // theta_ratio, 1)
        self.d_compressed = max(d_model // theta_ratio, 1)

        self.compress = nn.Sequential(
            nn.Linear(d_model, self.d_compressed),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.forecast_fc = nn.Linear(self.d_compressed, self.theta_size * n_quantiles)
        self.backcast_fc = nn.Linear(self.d_compressed, d_model)

    def forward(self, h):
        B, N, D = h.shape
        compressed = self.compress(h)                          # (B, N, d_compressed)

        theta_f = self.forecast_fc(compressed)                 # (B, N, theta_sz * nq)
        theta_f = theta_f.reshape(B, N, self.theta_size, self.n_quantiles)  # (B,N,theta_sz,nq)

        # Interpolate theta_size -> pred_len along dim=2
        if self.theta_size == self.pred_len:
            forecast = theta_f                                  # (B, N, S, nq)
        else:
            th = theta_f.permute(0, 1, 3, 2)                   # (B, N, nq, theta_sz)
            th = th.reshape(B * N * self.n_quantiles, 1, self.theta_size)
            f = F.interpolate(th, size=self.pred_len, mode='linear', align_corners=False)
            forecast = f.reshape(B, N, self.n_quantiles, self.pred_len)
            forecast = forecast.permute(0, 1, 3, 2)             # (B, N, S, nq)

        backcast = self.backcast_fc(compressed)                 # (B, N, D)
        return forecast, backcast


class ProbabilisticNHiTSHead(nn.Module):
    """
    Multi-scale head stacking 3 ProbabilisticNHiTSBlocks from coarse to fine.
    Output: (B, N, S, nq) — quantile forecasts per variate.
    """
    _THETA_RATIOS = [4, 2, 1]

    def __init__(self, d_model, pred_len, n_stacks=3, dropout=0.1, n_quantiles=99):
        super().__init__()
        self.pred_len = pred_len
        self.n_quantiles = n_quantiles
        self.n_stacks = n_stacks
        ratios = self._THETA_RATIOS[:n_stacks]
        self.blocks = nn.ModuleList([
            ProbabilisticNHiTSBlock(d_model, pred_len, r, dropout, n_quantiles)
            for r in ratios
        ])

    def forward(self, h):
        B, N, D = h.shape
        total = torch.zeros(B, N, self.pred_len, self.n_quantiles, device=h.device, dtype=h.dtype)
        residual = h
        for block in self.blocks:
            forecast, backcast = block(residual)
            total = total + forecast
            residual = residual - backcast
        return total  # (B, N, S, nq)


class iTransformerNHiTS_GEFCom(nn.Module):
    """
    Complete GEFCom model: iTransformer encoder + weather token injection + NHiTS quantile head.

    Input:  x_enc: (B, L, N)  — past window (N=50: 10 zones x 5 feats)
            x_future_weather: (B, 40) — U10/V10/U100/V100 x 10 zones
    Output: (B, N_targets, S=1, nq=99) — quantile predictions per target zone
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.enc_in = configs.enc_in

        self.quantiles = getattr(configs, 'quantiles', GEFCOM_QUANTILES)
        self.n_quantiles = len(self.quantiles)
        self.target_channels = getattr(configs, 'target_channels', None)
        self.n_targets = len(self.target_channels) if self.target_channels is not None else 10

        d_model = getattr(configs, 'd_model', 128)
        dropout = getattr(configs, 'dropout', 0.1)
        nhits_stacks = getattr(configs, 'nhits_n_stacks', 3)

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, d_model, configs.embed, configs.freq, dropout,
        )
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=dropout, output_attention=False),
                    d_model, configs.n_heads,
                ),
                d_model, configs.d_ff, dropout=dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )

        # NHiTS quantile head
        self.nhits_head = ProbabilisticNHiTSHead(
            d_model=d_model,
            pred_len=configs.pred_len,
            n_stacks=nhits_stacks,
            dropout=dropout,
            n_quantiles=self.n_quantiles,
        )

        # Per-target projection: NHiTS outputs (B, N, S, nq) for all variates
        # Target channels: select only the target variates
        self.target_channels = getattr(configs, 'target_channels', None)
        self.n_zones = configs.n_zones if hasattr(configs, 'n_zones') else 10

        # Weather injection: per-zone 4-var → D, added to TARGETVAR tokens
        self.weather_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # GEFCom: 5 feats per zone (TARGETVAR, U10, V10, U100, V100)
        # weather cols are the 4 non-TARGETVAR per zone
        self.register_buffer('weather_indices',
            torch.tensor([i for i in range(configs.enc_in) if i % 5 != 0], dtype=torch.long))

    def enforce_monotonicity(self, preds):
        base = preds[..., :1]
        deltas = F.softplus(preds[..., 1:] - preds[..., :-1])
        return torch.cumsum(torch.cat([base, deltas], dim=-1), dim=-1)

    def forward(self, x_enc, x_future_weather=None, x_mark_enc=None):
        """
        Args:
            x_enc: (B, L, N) — past window
            x_future_weather: (B, 40) — future U10/V10/U100/V100 x 10 zones, or None
            x_mark_enc: optional time features
        Returns:
            quantiles: (B, n_targets, S, nq)
        """
        B, L, N = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Encode past window
        enc_out = self.enc_embedding(x_enc, x_mark_enc)          # (B, N_data+N_time, D)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Select data variates only (exclude time tokens if any)
        n_data = self.enc_in
        enc_out = enc_out[:, :n_data, :]                          # (B, N_data, D)

        # Inject future weather INTO each TARGETVAR token's representation
        if x_future_weather is not None and self.target_channels is not None:
            if self.use_norm:
                w_means = means[:, 0, self.weather_indices]        # (B, 40)
                w_stds = stdev[:, 0, self.weather_indices]         # (B, 40)
                x_future_weather = (x_future_weather - w_means) / w_stds
            # Reshape (B,40) -> (B, n_zones, 4) per-zone weather
            w_per_zone = x_future_weather.reshape(B, self.n_zones, 4)
            w_emb = self.weather_proj(w_per_zone)                  # (B, n_zones, D)
            for zi, ti in enumerate(self.target_channels):
                enc_out[:, ti, :] = enc_out[:, ti, :] + w_emb[:, zi, :]

        # Probabilistic NHiTS head
        out = self.nhits_head(enc_out)                             # (B, N_data, S, nq)

        # Select target variates
        if self.target_channels is not None:
            out = out[:, self.target_channels, :, :]               # (B, n_targets, S, nq)

        out = self.enforce_monotonicity(out)                       # (B, n_targets, S, nq)

        # Denormalize each target channel
        if self.use_norm and self.target_channels is not None:
            for i, ci in enumerate(self.target_channels):
                s = stdev[:, 0, ci:ci+1]
                m = means[:, 0, ci:ci+1]
                out[:, i] = out[:, i] * s.unsqueeze(-1) + m.unsqueeze(-1)
        elif self.use_norm:
            out = out * stdev[:, 0, :].unsqueeze(-1).unsqueeze(-1) + means[:, 0, :].unsqueeze(-1).unsqueeze(-1)

        return out
