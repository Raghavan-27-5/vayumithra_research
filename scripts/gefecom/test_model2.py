import sys, numpy as np, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.probabilistic_model import iTransformerNHiTS_Probabilistic, PinballLoss, QUANTILES

class Cfg:
    seq_len, pred_len, enc_in = 336, 1, 51
    d_model, n_heads, e_layers = 128, 4, 2
    d_ff, dropout = 512, 0.05
    activation, embed, freq = "gelu", "timeF", "h"
    factor, class_strategy = 1, "projection"
    use_norm, output_attention = True, False

cfg = Cfg()
model = iTransformerNHiTS_Probabilistic(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n:,}")

x = torch.randn(4, 336, 51)
out = model(x_enc=x)
print(f"Output: {out.shape}")

assert out.shape == (4, 1, 4), f"Bad shape: {out.shape}"
assert (out[...,0] <= out[...,1]).all(), "P10 > P50"
assert (out[...,1] <= out[...,2]).all(), "P50 > P90"
print("Model OK")
