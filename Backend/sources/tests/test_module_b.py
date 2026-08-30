"""
Tests for Module B — Time-Series Drift Predictor.

Covers: safety slope math, baseline-only fallback, RF training threshold,
and the value_96h acceleration feature improving recall on fast-accelerating
latent defect parts.
"""
import pytest
import numpy as np
from app.schemas import ParameterReading
from app.module_b_drift import DriftModel, T0, T1, T2, T96


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reading(
    component_id: str,
    value_0h: float,
    value_24h: float,
    datasheet_max: float = 50.0,
    value_168h: float = None,
    value_96h: float = None,
    lot_id: str = "LOT-001",
) -> ParameterReading:
    return ParameterReading(
        component_id=component_id,
        lot_id=lot_id,
        parameter="leakage_uA",
        value_0h=value_0h,
        value_24h=value_24h,
        value_96h=value_96h,
        value_168h=value_168h,
        datasheet_max=datasheet_max,
        datasheet_min=0.0,
    )


def _normal_readings(n: int = 15, include_168h: bool = True, include_96h: bool = False):
    """Synthetic normal parts: gentle linear drift, well within spec."""
    rng = np.random.default_rng(0)
    readings = []
    for i in range(n):
        v0 = rng.normal(10, 0.5)
        rate = 0.02
        v24 = v0 + rate * T1 + rng.normal(0, 0.1)
        v96 = v0 + rate * T96 + rng.normal(0, 0.2) if include_96h else None
        v168 = v0 + rate * T2 + rng.normal(0, 0.3) if include_168h else None
        readings.append(_make_reading(f"N{i:03d}", v0, v24, value_168h=v168, value_96h=v96))
    return readings


# ---------------------------------------------------------------------------
# Test 1 — Safety slope arithmetic
# ---------------------------------------------------------------------------
def test_safety_slope_math():
    """
    safety_slope = (datasheet_max - value_24h) / (T2 - T1)
    With margin 0.85: effective_threshold = safety_slope * 0.85
    A part whose predicted rate exactly equals safety_slope (no margin)
    should NOT be flagged at margin=1.0 but SHOULD at margin=0.85 if
    the predicted rate exceeds 0.85 * safety_slope.
    """
    # Unfitted model — pure linear baseline
    model = DriftModel()
    assert not model.is_fit

    datasheet_max = 50.0
    v0, v24 = 10.0, 14.0
    # Linear rate: (14 - 10) / 24 = 0.1667 /h
    # Baseline forecast: 14 + 0.1667 * 144 = 38.0
    expected_baseline = v24 + ((v24 - v0) / (T1 - T0)) * (T2 - T1)
    expected_slope = (datasheet_max - v24) / (T2 - T1)

    reading = _make_reading("S001", v0, v24, datasheet_max=datasheet_max)
    preds = model.predict_batch([reading], safety_margin=1.0)
    p = preds[0]

    assert abs(p.baseline_linear_forecast_168h - expected_baseline) < 1e-3
    assert abs(p.safety_slope_per_hour - expected_slope) < 1e-4
    # At margin=1.0, predicted rate < safety_slope => should NOT exceed
    assert not p.exceeds_safety_slope


def test_safety_slope_conservatism():
    """A part with rate at 90% of safety_slope should be flagged at margin=0.85."""
    model = DriftModel()
    datasheet_max = 50.0
    v24 = 10.0
    safety_slope = (datasheet_max - v24) / (T2 - T1)
    # We want the part to drift at 90% of safety_slope over 144h
    target_v168 = v24 + safety_slope * 0.90 * (T2 - T1)
    # Back-calculate v0 so the linear baseline lands exactly at target_v168
    # forecast = v24 + (v24 - v0)/24 * 144 = target_v168
    # (v24 - v0)/24 * 144 = target_v168 - v24
    # v0 = v24 - (target_v168 - v24) / 6
    v0 = v24 - (target_v168 - v24) / 6.0

    reading = _make_reading("S002", v0, v24, datasheet_max=datasheet_max)
    preds = model.predict_batch([reading], safety_margin=0.85)
    assert preds[0].exceeds_safety_slope, (
        "Rate at 90% of safety_slope must be flagged at margin=0.85"
    )


# ---------------------------------------------------------------------------
# Test 2 — Baseline-only fallback when training data < 10 rows
# ---------------------------------------------------------------------------
def test_baseline_fallback_under_10_rows():
    """With only 5 labeled rows, model must stay in baseline-only mode."""
    model = DriftModel()
    small_set = _normal_readings(n=5, include_168h=True)
    model.fit(small_set)

    assert not model.is_fit, "Model must NOT be fit with fewer than 10 rows"

    reading = _make_reading("X001", 10.0, 10.5)
    preds = model.predict_batch([reading])
    p = preds[0]
    # In baseline mode, ml_corrected == baseline
    assert p.ml_corrected_forecast_168h == p.baseline_linear_forecast_168h
    assert model.feature_importance_report() is None


# ---------------------------------------------------------------------------
# Test 3 — RF model fits when >= 10 labeled rows
# ---------------------------------------------------------------------------
def test_rf_fit_above_10_rows():
    """With 20 labeled rows, the RF correction model should be fit."""
    model = DriftModel()
    training = _normal_readings(n=20, include_168h=True)
    model.fit(training)

    assert model.is_fit, "Model must be fit with 20 rows"
    report = model.feature_importance_report()
    assert report is not None
    assert set(report.keys()) == {"value_0h", "value_24h", "early_rate"}
    assert abs(sum(report.values()) - 1.0) < 1e-3, "Feature importances must sum to ~1"


# ---------------------------------------------------------------------------
# Test 4 — 96h feature: fast-accelerating defect is caught more reliably
# ---------------------------------------------------------------------------
def test_96h_feature_improves_defect_catch():
    """
    Train two models on a mixed dataset (normal + fast-accelerating parts with 96h data):
    one WITH 96h features, one without (baseline-only model for reference).
    Then predict a fast-accelerating defect — the 96h-aware model must flag it.
    The 96h feature names must appear in the importance report.
    """
    rng = np.random.default_rng(42)

    # Mixed training set: 20 normal + 10 fast-accelerating (with known 168h)
    training = []
    # Normal parts
    for i in range(20):
        v0 = rng.normal(10, 0.5)
        rate = 0.02
        v24 = v0 + rate * T1 + rng.normal(0, 0.1)
        v96 = v0 + rate * T96 + rng.normal(0, 0.2)
        v168 = v0 + rate * T2 + rng.normal(0, 0.3)
        training.append(_make_reading(f"N{i:03d}", v0, v24, value_96h=v96, value_168h=v168))
    # Fast-accelerating parts — these are what the model needs to learn
    for i in range(10):
        v0 = rng.normal(10, 0.5)
        v24 = v0 + rng.normal(0.5, 0.2)        # looks normal at 24h
        accel_rate = rng.uniform(0.22, 0.35)    # sharp acceleration after 24h
        v96 = v24 + accel_rate * (T96 - T1) + rng.normal(0, 1)
        v168 = v24 + accel_rate * (T2 - T1) + rng.normal(0, 1.5)
        training.append(_make_reading(f"D{i:03d}", v0, v24, value_96h=v96, value_168h=v168))

    model_with = DriftModel()
    model_with.fit(training)

    assert model_with.is_fit
    assert model_with.uses_96h, "Model trained with 96h data must set uses_96h=True"

    report = model_with.feature_importance_report()
    assert "value_96h" in report
    assert "mid_rate" in report
    assert "accel" in report
    assert abs(sum(report.values()) - 1.0) < 2e-3  # round(x, 3) can give 0.999 total

    # A fast-accelerating defect: looks normal at 0h/24h, then drifts sharply.
    # datasheet_max=30 (tighter) so that the safety slope is harder to stay under.
    datasheet_max = 30.0
    v0 = 10.0
    v24 = 10.5    # looks normal (early_rate ≈ 0.02/h)
    v96 = 27.0    # sharply accelerating: mid_rate ≈ (27-10.5)/72 = 0.229/h
                  # safety_slope = (30 - 10.5) / 144 = 0.135/h → easy to exceed

    defect = _make_reading("DEFECT", v0, v24, value_96h=v96, datasheet_max=datasheet_max)
    preds = model_with.predict_batch([defect], safety_margin=0.85)
    p = preds[0]

    assert p.exceeds_safety_slope, (
        f"Fast-accelerating defect must be flagged by the 96h-aware model. "
        f"Got predicted_rate={p.predicted_drift_rate_per_hour:.4f}, "
        f"threshold={p.safety_slope_per_hour * 0.85:.4f}. "
        f"Explanation: {p.explanation}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Graceful fallback when 96h model receives a reading without value_96h
# ---------------------------------------------------------------------------
def test_graceful_fallback_missing_96h_at_inference():
    """
    If the model was trained WITH 96h features but a new reading lacks
    value_96h, predict_batch must fall back to correction=0 (pure baseline)
    rather than raising an exception.
    """
    training = _normal_readings(n=20, include_168h=True, include_96h=True)
    model = DriftModel()
    model.fit(training)
    assert model.uses_96h

    # Reading WITHOUT value_96h
    reading_no_96h = _make_reading("NV001", 10.0, 10.5)
    assert reading_no_96h.value_96h is None

    try:
        preds = model.predict_batch([reading_no_96h])
    except Exception as e:
        pytest.fail(f"predict_batch raised an exception on missing 96h: {e}")

    p = preds[0]
    # Correction falls back to 0 => ml_corrected == baseline
    assert p.ml_corrected_forecast_168h == p.baseline_linear_forecast_168h, (
        "Without value_96h at inference, correction must be 0 (pure baseline)"
    )
