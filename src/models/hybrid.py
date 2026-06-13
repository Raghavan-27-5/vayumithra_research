"""
src/models/hybrid.py
─────────────────────
Hybrid ensemble models combining LGBM + DLinear/Mamba predictions.

Three strategies:
  1. WeightedAverage  — fixed or optimised linear blend
  2. ResidualCorrector — DL model corrects LGBM residuals
  3. StackedMetaLearner — Ridge regression on OOF predictions

All hybrids train ONLY on out-of-fold predictions.
No direct use of validation targets in training the meta-learner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class WeightedAverage:
    """
    Simple weighted average: pred = w*lgbm + (1-w)*dl
    Weight w is optimised on OOF predictions by minimising MAE.
    """
    def __init__(self):
        self.best_w: float = 0.5

    def fit(
        self,
        oof_lgbm: np.ndarray,
        oof_dl:   np.ndarray,
        y_oof:    np.ndarray,
    ) -> "WeightedAverage":
        best_mae = float("inf")
        for w in np.linspace(0, 1, 101):
            blended = w * oof_lgbm + (1 - w) * oof_dl
            mae = float(np.mean(np.abs(y_oof - blended)))
            if mae < best_mae:
                best_mae = mae
                self.best_w = w
        return self

    def predict(self, lgbm_preds: np.ndarray, dl_preds: np.ndarray) -> np.ndarray:
        return self.best_w * lgbm_preds + (1 - self.best_w) * dl_preds

    def summary(self) -> dict:
        return {"model": "weighted_avg", "lgbm_weight": self.best_w, "dl_weight": 1 - self.best_w}


class ResidualCorrector:
    """
    LGBM makes primary prediction.
    DL model corrects the residual: final = lgbm_pred + alpha * dl_residual_pred
    The alpha is fit on OOF to avoid over-correcting.
    """
    def __init__(self):
        self.alpha: float = 1.0

    def fit(
        self,
        oof_lgbm:   np.ndarray,
        oof_dl:     np.ndarray,
        y_oof:      np.ndarray,
    ) -> "ResidualCorrector":
        # The "dl_residual_pred" is: dl_pred - lgbm_pred
        # We want: y ≈ lgbm + alpha * (dl - lgbm)
        residuals   = oof_dl - oof_lgbm           # correction vector
        lgbm_errors = y_oof  - oof_lgbm            # what LGBM got wrong

        # Linear regression: lgbm_errors ≈ alpha * residuals
        if np.std(residuals) > 1e-8:
            self.alpha = float(
                np.dot(residuals, lgbm_errors) /
                (np.dot(residuals, residuals) + 1e-8)
            )
            self.alpha = float(np.clip(self.alpha, -2.0, 2.0))
        return self

    def predict(self, lgbm_preds: np.ndarray, dl_preds: np.ndarray) -> np.ndarray:
        correction = dl_preds - lgbm_preds
        return lgbm_preds + self.alpha * correction

    def summary(self) -> dict:
        return {"model": "residual_corrector", "alpha": self.alpha}


class StackedMetaLearner:
    """
    Ridge regression meta-learner trained on OOF predictions.
    Features: [lgbm_pred, dl_pred] → target: y
    Optionally includes additional OOF features.
    """
    def __init__(self, alpha: float = 1.0):
        self.ridge  = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(
        self,
        oof_lgbm:     np.ndarray,
        oof_dl:       np.ndarray,
        y_oof:        np.ndarray,
        extra_cols:   np.ndarray | None = None,   # additional OOF model preds
    ) -> "StackedMetaLearner":
        X_meta = np.column_stack([oof_lgbm, oof_dl])
        if extra_cols is not None:
            X_meta = np.column_stack([X_meta, extra_cols])

        X_scaled = self.scaler.fit_transform(X_meta)
        self.ridge.fit(X_scaled, y_oof)
        self._fitted = True
        return self

    def predict(
        self,
        lgbm_preds: np.ndarray,
        dl_preds:   np.ndarray,
        extra_cols: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        X_meta = np.column_stack([lgbm_preds, dl_preds])
        if extra_cols is not None:
            X_meta = np.column_stack([X_meta, extra_cols])
        X_scaled = self.scaler.transform(X_meta)
        return self.ridge.predict(X_scaled)

    def summary(self) -> dict:
        coef = self.ridge.coef_.tolist() if self._fitted else []
        return {
            "model":       "stacked_meta",
            "ridge_alpha": self.ridge.alpha,
            "coef":        coef,
            "intercept":   float(self.ridge.intercept_) if self._fitted else 0.0,
        }


def compare_hybrids(
    oof_lgbm: np.ndarray,
    oof_dl:   np.ndarray,
    y_oof:    np.ndarray,
    val_lgbm: np.ndarray,
    val_dl:   np.ndarray,
    y_val:    np.ndarray,
    horizon:  int,
    fold:     int,
) -> pd.DataFrame:
    """
    Fit all hybrid strategies on OOF, evaluate on val.
    Returns a comparison DataFrame.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def _metrics(y_true, y_pred, model_name) -> dict:
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        return {"model": model_name, "horizon": horizon, "fold": fold,
                "mae": mae, "rmse": rmse, "r2": r2}

    rows = []

    # Standalone baselines
    rows.append(_metrics(y_val, val_lgbm, "lgbm_standalone"))
    rows.append(_metrics(y_val, val_dl,   "dl_standalone"))

    # Weighted average
    wa = WeightedAverage().fit(oof_lgbm, oof_dl, y_oof)
    rows.append({**_metrics(y_val, wa.predict(val_lgbm, val_dl), "weighted_avg"),
                 **wa.summary()})

    # Residual corrector
    rc = ResidualCorrector().fit(oof_lgbm, oof_dl, y_oof)
    rows.append({**_metrics(y_val, rc.predict(val_lgbm, val_dl), "residual_corrector"),
                 **rc.summary()})

    # Stacked meta-learner
    ml = StackedMetaLearner(alpha=1.0).fit(oof_lgbm, oof_dl, y_oof)
    rows.append({**_metrics(y_val, ml.predict(val_lgbm, val_dl), "stacked_meta"),
                 **ml.summary()})

    return pd.DataFrame(rows)
