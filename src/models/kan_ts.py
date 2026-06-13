"""src/models/kan_ts.py — KAN (Kolmogorov-Arnold Networks) time series forecaster.

Uses pykan (Liu et al. 2024): pip install pykan
KAN replaces standard linear layers with learnable spline functions,
allowing each "weight" to be a univariate nonlinear function.

For wind forecasting we use KAN as the forecasting head replacing the
linear decoder of the DLinear or patch-based encoder.
"""
import torch
import torch.nn as nn

try:
    from kan import KAN
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    print("[WARN] pykan not installed — KAN models unavailable. pip install pykan")


class KANForecaster(nn.Module):
    """
    Minimal KAN-based wind speed forecaster.

    Architecture:
      1. Flatten look-back window into a feature vector
      2. KAN network → learnable univariate splines per connection
      3. Output: pred_len values per variate

    This is intentionally the simplest direct-mapping KAN.
    It can be extended with patch embeddings or convolution front-ends.

    Args:
        seq_len   : look-back window L
        pred_len  : forecast horizon T
        enc_in    : number of input variates C (treated channel-independently)
        hidden    : KAN hidden layer width list, e.g. [64, 32]
        grid      : number of spline grid points (higher = more expressive)
        k         : spline order (default 3 = cubic)
    """

    def __init__(
        self,
        seq_len:  int,
        pred_len: int,
        enc_in:   int       = 1,
        hidden:   list[int] = None,
        grid:     int       = 5,
        k:        int       = 3,
    ):
        super().__init__()
        self.enc_in  = enc_in
        self.pred_len = pred_len

        if not KAN_AVAILABLE:
            raise ImportError("Install pykan: pip install pykan")

        hidden = hidden or [64, 32]
        # KAN takes a flat layer-width list: [input, *hidden, output]
        layers = [seq_len] + hidden + [pred_len]

        # One KAN per variate (channel-independent, matches DLinear philosophy)
        self.kans = nn.ModuleList([
            KAN(width=layers, grid=grid, k=k, seed=42)
            for _ in range(enc_in)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
        Returns:
            (B, T, C)
        """
        outs = []
        for c in range(self.enc_in):
            xc   = x[:, :, c]                      # (B, L)
            outs.append(self.kans[c](xc))           # (B, T)
        return torch.stack(outs, dim=-1)            # (B, T, C)


class LinearKANForecaster(nn.Module):
    """
    Hybrid: DLinear-style decomposition front-end + KAN decoder.
    Captures trend/seasonality inductive bias AND non-linear spline mapping.
    """

    def __init__(self, seq_len: int, pred_len: int, enc_in: int = 1,
                 kernel_size: int = 25, hidden: list[int] = None,
                 grid: int = 5, k: int = 3):
        super().__init__()
        from src.models.dlinear import SeriesDecomposition
        self.decomp = SeriesDecomposition(kernel_size)
        hidden = hidden or [32]

        if not KAN_AVAILABLE:
            raise ImportError("Install pykan: pip install pykan")

        layers = [seq_len] + hidden + [pred_len]
        self.kan_seasonal = nn.ModuleList([KAN(layers, grid, k) for _ in range(enc_in)])
        self.kan_trend    = nn.ModuleList([KAN(layers, grid, k) for _ in range(enc_in)])
        self.enc_in   = enc_in
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)    # each (B, L, C)
        outs = []
        for c in range(self.enc_in):
            out_s = self.kan_seasonal[c](seasonal[:, :, c])
            out_t = self.kan_trend[c](trend[:, :, c])
            outs.append(out_s + out_t)
        return torch.stack(outs, dim=-1)    # (B, T, C)
