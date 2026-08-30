"""
Module A — Dynamic Outlier Detection

Core idea (from the problem statement):
    A part at 45uA is a massive anomaly if its LOT averages 10uA, even
    though the datasheet's absolute ceiling is 50uA. Static limits alone
    would let it through.

Design choices, and why:

1. Robust statistics, not mean/std.
   If we used the lot mean and standard deviation, a handful of true
   latent-defect parts would inflate the std and mask themselves (and
   each other). We use the MEDIAN and MAD (Median Absolute Deviation)
   instead — these are robust to the very outliers we are trying to
   catch (up to 50% breakdown point).

2. Modified Z-score (Iglewicz & Hoaglin).
       M_i = 0.6745 * (x_i - median) / MAD
   Threshold of 3.5 is the standard recommendation. This is the metric
   that answers "how anomalous is this part relative to its own lot".

3. Hard datasheet ceiling as a non-negotiable safety net.
   Because false negatives are catastrophic per the evaluation rubric,
   ANY value beyond the absolute datasheet max/min is an automatic
   reject regardless of what the lot statistics say. Dynamic detection
   only ever ADDS sensitivity, it never removes the static check.

4. Severity bands instead of a binary flag.
   "watch" vs "reject" gives QA a triage lane instead of a black-box
   pass/fail, which also supports the Explainability requirement.

5. Every flag carries a human-readable explanation string built from
   the actual numbers used in the decision — this is what QA needs to
   see instead of a bare classification.
"""

import numpy as np
import pandas as pd
from typing import List
from app.schemas import ParameterReading, OutlierFlag

MAD_SCALE = 0.6745  # makes MAD comparable to std under normality assumption


def _robust_stats(values: np.ndarray):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    # Guard against a degenerate lot where every part reads identically
    # (MAD = 0 would cause a divide-by-zero / infinite z-score).
    if mad == 0:
        mad = 1e-6
    return median, mad


def detect_outliers(
    readings: List[ParameterReading],
    checkpoint: str = "value_24h",
    z_threshold: float = 3.5,
) -> List[OutlierFlag]:
    """
    Detect dynamic, lot-relative outliers at a given checkpoint
    (default: 24h, the earliest point where lot statistics are already
    meaningful and early rejection is still cheap).
    """
    df = pd.DataFrame([r.model_dump() for r in readings])
    results: List[OutlierFlag] = []

    # Group by (lot_id, parameter): different parameters have different
    # units/scales, so they must never be pooled together statistically.
    for (lot_id, parameter), group in df.groupby(["lot_id", "parameter"]):
        values = group[checkpoint].astype(float).to_numpy()
        median, mad = _robust_stats(values)

        for _, row in group.iterrows():
            value = float(row[checkpoint])
            z = MAD_SCALE * (value - median) / mad

            exceeds_limit = value > row["datasheet_max"] or value < row.get("datasheet_min", 0.0)
            is_dynamic_outlier = abs(z) > z_threshold

            if exceeds_limit:
                severity = "reject"
                is_anomalous = True
                explanation = (
                    f"HARD LIMIT VIOLATION: {parameter}={value:.3f} exceeds datasheet "
                    f"max {row['datasheet_max']:.3f} (lot {lot_id} median={median:.3f})."
                )
            elif is_dynamic_outlier:
                # Distinguish "clearly bad" from "borderline" for QA triage
                severity = "reject" if abs(z) > z_threshold * 1.5 else "watch"
                is_anomalous = True
                explanation = (
                    f"DYNAMIC OUTLIER: {parameter}={value:.3f} vs lot {lot_id} median="
                    f"{median:.3f} (MAD={mad:.3f}) -> modified z-score={z:.2f} "
                    f"(threshold={z_threshold}). Within datasheet limit "
                    f"({row['datasheet_max']:.3f}) but statistically anomalous "
                    f"for this lot — classic latent-defect signature."
                )
            else:
                severity = "none"
                is_anomalous = False
                explanation = (
                    f"Nominal: {parameter}={value:.3f}, z-score={z:.2f}, within "
                    f"lot {lot_id} normal range and datasheet limit."
                )

            results.append(
                OutlierFlag(
                    component_id=row["component_id"],
                    lot_id=lot_id,
                    parameter=parameter,
                    checkpoint=checkpoint,
                    value=value,
                    lot_median=median,
                    lot_mad=mad,
                    modified_z_score=round(z, 3),
                    exceeds_datasheet_limit=exceeds_limit,
                    is_anomalous=is_anomalous,
                    severity=severity,
                    explanation=explanation,
                )
            )

    return results
