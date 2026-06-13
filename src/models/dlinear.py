"""src/models/dlinear.py — DLinear model (Zeng et al. 2022, arXiv:2205.13504).

Architecture:
  1. Decompose input into trend (moving average) + seasonal (residual)
  2. Two independent linear layers (one per branch)
  3. Sum outputs

Supports both channel-independent (CI) and channel-dependent (CD) modes.
"""
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average for trend extraction. Pads both ends to preserve length."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)  →  pad along time axis
        front = x[:, :1, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)          # (B, L + pad, C)
        x = self.avg(x.permute(0, 2, 1))               # (B, C, L)
        return x.permute(0, 2, 1)                      # (B, L, C)


class SeriesDecomposition(nn.Module):
    """Decomposes series into trend + seasonal."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear model.

    Args:
        seq_len    : look-back window length (L)
        pred_len   : forecast horizon (T)
        enc_in     : number of input variates (C)
        kernel_size: moving average kernel size (default 25)
        individual : if True, each variate gets its own weight pair (CI mode)
    """

    def __init__(
        self,
        seq_len:     int,
        pred_len:    int,
        enc_in:      int  = 1,
        kernel_size: int  = 25,
        individual:  bool = True,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.enc_in     = enc_in
        self.individual = individual

        self.decomp = SeriesDecomposition(kernel_size)

        if individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend    = nn.Linear(seq_len, pred_len)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights to 1/L for interpretability (as in paper)."""
        if self.individual:
            for layer in self.Linear_Seasonal:
                nn.init.constant_(layer.weight, 1.0 / self.seq_len)
                nn.init.zeros_(layer.bias)
            for layer in self.Linear_Trend:
                nn.init.constant_(layer.weight, 1.0 / self.seq_len)
                nn.init.zeros_(layer.bias)
        else:
            nn.init.constant_(self.Linear_Seasonal.weight, 1.0 / self.seq_len)
            nn.init.zeros_(self.Linear_Seasonal.bias)
            nn.init.constant_(self.Linear_Trend.weight, 1.0 / self.seq_len)
            nn.init.zeros_(self.Linear_Trend.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)  — batch of look-back windows

        Returns:
            (B, T, C) — multi-step forecast
        """
        seasonal, trend = self.decomp(x)   # each: (B, L, C)

        if self.individual:
            out_s = torch.stack(
                [self.Linear_Seasonal[c](seasonal[:, :, c]) for c in range(self.enc_in)],
                dim=-1,
            )  # (B, T, C)
            out_t = torch.stack(
                [self.Linear_Trend[c](trend[:, :, c]) for c in range(self.enc_in)],
                dim=-1,
            )
        else:
            # (B, L, C) → permute to (B, C, L) → linear → (B, C, T) → permute
            out_s = self.Linear_Seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            out_t = self.Linear_Trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        return out_s + out_t   # (B, T, C)


class NLinear(nn.Module):
    """
    NLinear: Normalized Linear model. Handles distribution shift by subtracting
    the last value of the look-back window before projection.
    """

    def __init__(self, seq_len: int, pred_len: int, enc_in: int = 1, individual: bool = True):
        super().__init__()
        self.individual = individual
        if individual:
            self.Linear = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
        else:
            self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :].detach()       # (B, 1, C) — local shift
        x    = x - last
        if self.individual:
            out = torch.stack(
                [self.Linear[c](x[:, :, c]) for c in range(x.size(-1))], dim=-1
            )
        else:
            out = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out + last                  # add back the shift
