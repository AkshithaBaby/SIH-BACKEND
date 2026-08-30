"""
Module B — Time-Series Drift Predictor

Task: given Value_0h and Value_24h, forecast Value_168h, then compare the
IMPLIED drift rate against a "safety slope" derived from the datasheet
limit. If a part is drifting fast enough that it would blow through spec
by 168h, reject it early — don't wait 168h to find out.

Design, and why it's built this way rather than as a pure black-box model:

1. Physics-informed baseline (explainable by construction).
   Most burn-in degradation (leakage growth, threshold shift) is
   approximately linear or log-linear over short windows. So the
   baseline forecast is simple rate extrapolation:
       rate_early = (Value_24h - Value_0h) / 24
       forecast_168h = Value_24h + rate_early * (168 - 24)
   This alone is fully explainable to a QA inspector — it's just
   "if it keeps drifting at the rate it's already shown, here's where
   it lands." No black box required for this part.

2. ML correction layer (learns systematic bias the linear model misses).
   Real drift is often slightly supra-linear (e.g. leakage current in
   semiconductors tends to follow a power-law / accelerating trend
   under sustained thermal stress). A small RandomForestRegressor is
   trained on historical lots (rows WITH known 168h ground truth) to
   predict the RESIDUAL between the true 168h value and the linear
   baseline, using [value_0h, value_24h, early_rate] as features, plus
   [value_96h, mid_rate, accel] when a 96h mid-point reading is available.
   Its correction is reported explicitly, and feature_importances_ are
   surfaced so QA can see what's driving the correction — this keeps
   the "black box" part small, bounded, and inspectable rather than
   being the whole prediction.
   If no historical training data is available yet, the system
   gracefully falls back to the pure explainable linear baseline
   (correction = 0), so it never silently depends on a model that
   doesn't exist.

3. Optional value_96h feature (Task 3 extension).
   When a mid-point 96h reading is available, three extra features are
   added to the RF correction model:
       mid_rate  = (value_96h - value_24h) / (96 - 24)   [24h→96h rate]
       accel     = mid_rate - early_rate                  [acceleration signal]
   A fast-accelerating part has accel >> 0, giving the RF a strong signal
   that the linear extrapolation from 0h/24h will underestimate 168h.
   Backward-compatible: if value_96h was absent at training time,
   prediction also uses the 3-feature set; if present at training time,
   it is required at prediction time (stored in self.uses_96h).

4. Safety slope.
       safety_slope = (datasheet_max - Value_24h) / (168 - 24)
   This is the maximum rate of change the part can sustain from 24h
   onward and still land AT the limit by 168h. A configurable
   `safety_margin` (<1.0) tightens this further, trading a few more
   false positives for fewer false negatives — appropriate given the
   rubric's emphasis on catastrophic cost of missed defects.

5. MAE tracking.
   When actual_value_168h is supplied (e.g. in a backtest/validation
   batch), absolute_error is computed per-part so the batch-level MAE
   metric required by the rubric can be aggregated by the caller.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.schemas import ParameterReading, DriftPrediction

T0, T1, T96, T2 = 0.0, 24.0, 96.0, 168.0


def _linear_baseline(value_0h: float, value_24h: float) -> tuple[float, float]:
    rate = (value_24h - value_0h) / (T1 - T0)
    forecast = value_24h + rate * (T2 - T1)
    return forecast, rate


class DriftModel:
    """
    Wraps the optional ML correction layer. Call `.fit()` with historical
    readings that include value_168h ground truth (e.g. from prior lots
    or a held-out training split) before using `.predict_batch()` for
    the ML-corrected forecast. If never fit, predictions fall back to
    the pure explainable linear baseline.

    When `value_96h` is present in training data, the model uses a
    6-feature set (adding mid_rate and acceleration), giving much better
    recall on fast-accelerating latent defects. This choice is stored in
    `self.uses_96h` and must match at prediction time.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=4, random_state=random_state
        )
        self.is_fit = False
        self.uses_96h = False
        self._base_feature_names = ["value_0h", "value_24h", "early_rate"]
        self._96h_feature_names = ["value_96h", "mid_rate", "accel"]

    @property
    def feature_names(self) -> list[str]:
        if self.uses_96h:
            return self._base_feature_names + self._96h_feature_names
        return self._base_feature_names

    def _build_features(self, df: pd.DataFrame, has_96h: bool) -> np.ndarray:
        """Build the feature matrix from a DataFrame of readings."""
        early_rate = (df["value_24h"] - df["value_0h"]) / (T1 - T0)
        base = np.column_stack([df["value_0h"], df["value_24h"], early_rate])

        if has_96h:
            mid_rate = (df["value_96h"] - df["value_24h"]) / (T96 - T1)
            accel = mid_rate - early_rate
            return np.column_stack([base, df["value_96h"], mid_rate, accel])

        return base

    def fit(self, historical: List[ParameterReading]) -> DriftModel:
        df = pd.DataFrame([r.model_dump() for r in historical])
        df = df.dropna(subset=["value_168h"])

        if len(df) < 10:
            # Not enough data to train a reliable correction model —
            # explicitly stay in baseline-only mode rather than
            # overfitting to a handful of points.
            self.is_fit = False
            return self

        # Use 96h features if available in the majority of training rows.
        has_96h = bool("value_96h" in df.columns and df["value_96h"].notna().mean() >= 0.5)
        self.uses_96h = has_96h

        X = self._build_features(df, has_96h)

        baseline_forecast = df["value_24h"] + (
            (df["value_24h"] - df["value_0h"]) / (T1 - T0)
        ) * (T2 - T1)
        residual = df["value_168h"].to_numpy() - baseline_forecast.to_numpy()

        self.model.fit(X, residual)
        self.is_fit = True
        return self

    def feature_importance_report(self) -> Optional[dict]:
        if not self.is_fit:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_.round(3)))

    def predict_batch(
        self,
        readings: List[ParameterReading],
        safety_margin: float = 0.85,
    ) -> List[DriftPrediction]:
        results = []
        for r in readings:
            baseline_forecast, early_rate = _linear_baseline(r.value_0h, r.value_24h)

            if self.is_fit:
                row_df = pd.DataFrame([r.model_dump()])
                # At prediction time, use 96h features only when the model
                # was trained with them AND the reading actually has value_96h.
                predict_with_96h = self.uses_96h and r.value_96h is not None
                x = self._build_features(row_df, predict_with_96h)

                # If the model expects 96h features but reading lacks them,
                # fall back to a zero-correction baseline rather than crashing.
                if self.uses_96h and not predict_with_96h:
                    correction = 0.0
                    correction_note = (
                        ", no ML correction applied (model trained with 96h features "
                        "but value_96h absent — using pure physics baseline)"
                    )
                else:
                    correction = float(self.model.predict(x)[0])
                    if self.uses_96h:
                        mid_rate = (r.value_96h - r.value_24h) / (T96 - T1)
                        accel = mid_rate - early_rate
                        correction_note = (
                            f", ML correction = {correction:+.3f} "
                            f"[96h mid-rate={mid_rate:.4f}/h, accel={accel:.4f}/h] "
                            f"(trained on historical lots)"
                        )
                    else:
                        correction_note = (
                            f", ML correction = {correction:+.3f} (trained on historical lots)"
                        )
            else:
                correction = 0.0
                correction_note = (
                    ", no ML correction applied "
                    "(insufficient training history, using pure physics baseline)"
                )

            ml_forecast = baseline_forecast + correction

            safety_slope = (r.datasheet_max - r.value_24h) / (T2 - T1)
            effective_threshold = safety_slope * safety_margin
            predicted_rate = (ml_forecast - r.value_24h) / (T2 - T1)
            exceeds = predicted_rate > effective_threshold

            actual_error = None
            if r.value_168h is not None:
                actual_error = round(abs(ml_forecast - r.value_168h), 4)

            explanation = (
                f"Early drift rate (0h->24h) = {early_rate:.4f}/h. Linear baseline "
                f"forecast for 168h = {baseline_forecast:.3f}"
                + correction_note
                + f" -> final forecast = {ml_forecast:.3f}. "
                f"Safety slope = ({r.datasheet_max:.3f} - {r.value_24h:.3f}) / 144h = "
                f"{safety_slope:.4f}/h; with {int((1 - safety_margin) * 100)}% conservatism "
                f"margin, threshold = {effective_threshold:.4f}/h. "
                f"Predicted rate {predicted_rate:.4f}/h "
                f"{'EXCEEDS' if exceeds else 'is within'} the safety threshold."
            )

            results.append(
                DriftPrediction(
                    component_id=r.component_id,
                    lot_id=r.lot_id,
                    parameter=r.parameter,
                    value_0h=r.value_0h,
                    value_24h=r.value_24h,
                    baseline_linear_forecast_168h=round(baseline_forecast, 4),
                    ml_corrected_forecast_168h=round(ml_forecast, 4),
                    predicted_drift_rate_per_hour=round(predicted_rate, 5),
                    safety_slope_per_hour=round(safety_slope, 5),
                    exceeds_safety_slope=exceeds,
                    predicted_value_168h=round(ml_forecast, 4),
                    actual_value_168h=r.value_168h,
                    absolute_error=actual_error,
                    explanation=explanation,
                )
            )
        return results
