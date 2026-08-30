"""
Combines Module A (dynamic outlier detection) and Module B (drift
prediction) into a single QA-facing verdict per component/parameter.

Decision logic (conservative by design, per "false negatives are
catastrophic"):
    REJECT  if Module A says "reject"  OR Module B exceeds safety slope
    WATCH   if Module A says "watch"   (borderline lot-relative anomaly)
    PASS    otherwise

Either module can independently trigger REJECT — they are not averaged
or voted, because averaging could dilute a strong single-module signal
and let a genuine defect through.
"""
from __future__ import annotations

from typing import List
import numpy as np

from app.schemas import (
    ParameterReading,
    ScreeningVerdict,
    ModelMetrics,
)
from app.module_a_outlier import detect_outliers
from app.module_b_drift import DriftModel


def run_screening(
    readings: List[ParameterReading],
    drift_model: DriftModel,
    z_threshold: float = 3.5,
    safety_margin: float = 0.85,
) -> tuple[List[ScreeningVerdict], ModelMetrics]:

    outlier_flags = detect_outliers(readings, checkpoint="value_24h", z_threshold=z_threshold)
    drift_preds = drift_model.predict_batch(readings, safety_margin=safety_margin)

    outlier_by_key = {(f.component_id, f.parameter): f for f in outlier_flags}
    drift_by_key = {(d.component_id, d.parameter): d for d in drift_preds}

    verdicts: List[ScreeningVerdict] = []
    errors = []

    for r in readings:
        key = (r.component_id, r.parameter)
        o = outlier_by_key[key]
        d = drift_by_key[key]

        if o.severity == "reject" or d.exceeds_safety_slope:
            decision = "REJECT"
        elif o.severity == "watch":
            decision = "WATCH"
        else:
            decision = "PASS"

        qa_summary = (
            f"[{decision}] {r.component_id} / {r.parameter} (lot {r.lot_id}). "
            f"Module A: {o.explanation} | Module B: {d.explanation}"
        )

        verdicts.append(
            ScreeningVerdict(
                component_id=r.component_id,
                lot_id=r.lot_id,
                parameter=r.parameter,
                final_decision=decision,
                outlier_result=o,
                drift_result=d,
                qa_summary=qa_summary,
            )
        )

        if d.absolute_error is not None:
            errors.append(d.absolute_error)

    metrics = ModelMetrics(
        mae_168h=round(float(np.mean(errors)), 4) if errors else None,
        n_components=len(readings),
        n_flagged_module_a=sum(1 for o in outlier_flags if o.is_anomalous),
        n_flagged_module_b=sum(1 for d in drift_preds if d.exceeds_safety_slope),
        n_flagged_either=sum(1 for v in verdicts if v.final_decision in ("REJECT", "WATCH")),
    )

    return verdicts, metrics
