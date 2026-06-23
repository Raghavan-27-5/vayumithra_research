import sys, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.probabilistic_model import (
    iTransformerNHiTS_Probabilistic,
    PinballLoss,
    enforce_quantile_monotonicity,
    QUANTILES,
)
print("All imports OK")
print("Quantiles:", QUANTILES)

# Test instantiation
class Cfg:
    seq_len = 336
    pred_len = 1
    enc_in = 50
    d_model = 256
    n_heads = 4
    e_layers = 2
    d_ff = 1024
    dropout = 0.1
    activation = "gelu"
    embed = "timeF"
    freq = "h"
    factor = 1
    class_strategy = "projection"
    use_norm = True
    output_attention = False

cfg = Cfg()
model = iTransformerNHiTS_Probabilistic(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"Model params: {n:,}")

# Test forward pass
x = torch.randn(4, 336, 50)
with torch.no_grad():
    out = model(x_enc=x)
print(f"Output shape: {out.shape}")  # Expected: (4, 1, 4) = (B, S, n_quantiles)

# Test monotonicity
assert (out[..., 0] <= out[..., 1]).all(), "P10 > P50"
assert (out[..., 1] <= out[..., 2]).all(), "P50 > P90"
assert (out[..., 2] <= out[..., 3]).all(), "P90 > P99"
print("Monotonicity OK")

# Test PinballLoss
criterion = PinballLoss(quantiles=QUANTILES)
loss = criterion(out, torch.randn(4, 1))
print(f"PinballLoss: {loss.item():.4f}")

print("\n✅ Model ready for training")
