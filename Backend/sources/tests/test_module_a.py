"""
Tests for Module A — Dynamic Outlier Detection.

We build small, deterministic synthetic lots so the expected behaviour
is computed by hand alongside the test, making the assertions
self-documenting.
"""
import pytest
from app.schemas import ParameterReading
from app.module_a_outlier import detect_outliers, MAD_SCALE


def _make_reading(
    component_id: str,
    lot_id: str = "LOT-001",
    parameter: str = "leakage_uA",
    value_24h: float = 10.0,
    datasheet_max: float = 50.0,
    value_0h: float = 9.0,
) -> ParameterReading:
    return ParameterReading(
        component_id=component_id,
        lot_id=lot_id,
        parameter=parameter,
        value_0h=value_0h,
        value_24h=value_24h,
        datasheet_max=datasheet_max,
        datasheet_min=0.0,
    )


# ---------------------------------------------------------------------------
# Test 1 — A value 4× the lot median should be flagged as anomalous
# ---------------------------------------------------------------------------
def test_known_outlier_flagged():
    """
    Lot of 9 normal parts at ~10 uA + one outlier at 40 uA.
    With lot median ≈ 10, MAD ≈ 0, the outlier z-score >> 3.5.
    """
    readings = [_make_reading(f"C{i:03d}", value_24h=10.0 + i * 0.05) for i in range(9)]
    # One clear outlier — 4× the lot typical value
    readings.append(_make_reading("C999", value_24h=40.0))

    flags = detect_outliers(readings, checkpoint="value_24h", z_threshold=3.5)
    flag_map = {f.component_id: f for f in flags}

    outlier_flag = flag_map["C999"]
    assert outlier_flag.is_anomalous, "Outlier at 4× lot median must be flagged"
    assert outlier_flag.severity in ("watch", "reject"), "Severity must be watch or reject"
    assert outlier_flag.modified_z_score > 3.5, "z-score must exceed threshold"


# ---------------------------------------------------------------------------
# Test 2 — Tightly clustered lot: no part should be flagged
# ---------------------------------------------------------------------------
def test_known_normal_not_flagged():
    """
    10 parts with values differing by < 0.1 uA — all well within any
    reasonable z-score threshold.
    """
    readings = [_make_reading(f"C{i:03d}", value_24h=10.0 + i * 0.02) for i in range(10)]

    flags = detect_outliers(readings, checkpoint="value_24h", z_threshold=3.5)
    for f in flags:
        assert not f.is_anomalous, f"{f.component_id} should NOT be flagged in a tight cluster"
        assert f.severity == "none"


# ---------------------------------------------------------------------------
# Test 3 — Hard datasheet limit is always enforced regardless of z-score
# ---------------------------------------------------------------------------
def test_hard_limit_always_rejects():
    """
    A value that exceeds datasheet_max must be flagged even if the lot
    z-score wouldn't flag it (e.g. if every part is similarly high).
    """
    # All parts at 55 uA — datasheet max is 50 — MAD ≈ 0, z ≈ 0
    readings = [_make_reading(f"C{i:03d}", value_24h=55.0, datasheet_max=50.0) for i in range(5)]

    flags = detect_outliers(readings, checkpoint="value_24h", z_threshold=3.5)
    for f in flags:
        assert f.is_anomalous, "Value above datasheet_max must always be flagged"
        assert f.exceeds_datasheet_limit
        assert f.severity == "reject"


# ---------------------------------------------------------------------------
# Test 4 — Degenerate lot (all identical values) must not crash
# ---------------------------------------------------------------------------
def test_degenerate_lot_no_crash():
    """
    When all parts in a lot have the identical reading, MAD = 0.
    The guard (mad = 1e-6) must prevent divide-by-zero; the function
    must return without raising and normal parts must not be flagged.
    """
    readings = [_make_reading(f"C{i:03d}", value_24h=12.345) for i in range(8)]

    try:
        flags = detect_outliers(readings, checkpoint="value_24h", z_threshold=3.5)
    except ZeroDivisionError:
        pytest.fail("detect_outliers raised ZeroDivisionError on degenerate lot (MAD=0)")

    # All identical and within limit — none should be anomalous
    for f in flags:
        assert not f.is_anomalous


# ---------------------------------------------------------------------------
# Test 5 — Severity bands: just-above threshold -> "watch", far above -> "reject"
# ---------------------------------------------------------------------------
def test_severity_bands():
    """
    z just above 3.5 (< 5.25 = 3.5*1.5) => "watch"
    z far above 5.25                      => "reject"

    We use 30 tightly clustered background parts so the MAD is stable and
    is not shifted by adding the 2 outlier parts. Then we place outliers at
    exactly the desired z-score by computing values from the observed lot stats.
    """
    import numpy as np

    # 30 tightly clustered background parts: median ≈ 10.0, MAD ≈ 0.02
    n_bg = 30
    readings = [
        _make_reading(f"N{i:02d}", value_24h=10.0 + (i % 3 - 1) * 0.02) for i in range(n_bg)
    ]

    # Compute actual lot stats from the background
    bg_vals = np.array([r.value_24h for r in readings])
    median = float(np.median(bg_vals))
    mad = float(np.median(np.abs(bg_vals - median)))
    if mad == 0:
        mad = 1e-6

    # Place BORDER at z = 3.6 (just above 3.5, well below 5.25)
    border_val = median + (3.6 / MAD_SCALE) * mad
    # Place REJECT at z = 6.0 (well above 5.25 = 3.5 * 1.5)
    reject_val = median + (6.0 / MAD_SCALE) * mad

    readings.append(_make_reading("BORDER", value_24h=round(border_val, 6)))
    readings.append(_make_reading("REJECT", value_24h=round(reject_val, 6)))

    flags = detect_outliers(readings, checkpoint="value_24h", z_threshold=3.5)
    flag_map = {f.component_id: f for f in flags}

    assert flag_map["BORDER"].is_anomalous, "BORDER must be flagged as anomalous"
    assert flag_map["BORDER"].severity == "watch", "Just-above-threshold should be 'watch'"
    assert flag_map["REJECT"].severity == "reject", "Far-above-threshold should be 'reject'"
