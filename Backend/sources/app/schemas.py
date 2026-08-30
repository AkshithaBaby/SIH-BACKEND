"""
Pydantic data models for the Burn-In / ESS predictive screening backend.

Design note:
Each "reading" is one parameter (e.g. Iddq, leakage, propagation delay) on one
component, measured at fixed time checkpoints. Components are grouped by
lot_id because Module A's dynamic outlier logic is defined RELATIVE to the
lot's own distribution, not a global constant.
"""
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


class ParameterReading(BaseModel):
    component_id: str
    lot_id: str
    parameter: str = Field(..., description="e.g. 'Iddq', 'leakage_current', 'prop_delay'")
    value_0h: float
    value_24h: float
    value_96h: Optional[float] = None
    value_168h: Optional[float] = Field(
        None,
        description="Ground truth if known (e.g. during training/backtest). Omit at inference time.",
    )
    datasheet_max: float = Field(..., description="Absolute static limit — the hard safety ceiling.")
    datasheet_min: Optional[float] = 0.0


class OutlierFlag(BaseModel):
    component_id: str
    lot_id: str
    parameter: str
    checkpoint: str
    value: float
    lot_median: float
    lot_mad: float
    modified_z_score: float
    exceeds_datasheet_limit: bool
    is_anomalous: bool
    severity: str  # "none" | "watch" | "reject"
    explanation: str


class DriftPrediction(BaseModel):
    component_id: str
    lot_id: str
    parameter: str
    value_0h: float
    value_24h: float
    baseline_linear_forecast_168h: float
    ml_corrected_forecast_168h: float
    predicted_drift_rate_per_hour: float
    safety_slope_per_hour: float
    exceeds_safety_slope: bool
    predicted_value_168h: float
    actual_value_168h: Optional[float] = None
    absolute_error: Optional[float] = None
    explanation: str


class ScreeningVerdict(BaseModel):
    component_id: str
    lot_id: str
    parameter: str
    final_decision: str  # "PASS" | "WATCH" | "REJECT"
    outlier_result: OutlierFlag
    drift_result: DriftPrediction
    qa_summary: str


class BatchRequest(BaseModel):
    readings: List[ParameterReading]
    z_score_threshold: float = Field(
        3.5, description="Modified z-score threshold (Iglewicz & Hoaglin default)."
    )
    safety_margin: float = Field(
        0.85,
        description=(
            "Fraction of safety_slope allowed before flagging "
            "(e.g. 0.85 = flag at 85% of the limit, for conservatism against false negatives)."
        ),
    )


class ModelMetrics(BaseModel):
    mae_168h: Optional[float] = None
    n_components: int
    n_flagged_module_a: int
    n_flagged_module_b: int
    n_flagged_either: int


class ScreenBatchResponse(BaseModel):
    """Combined response from /screen endpoint."""

    verdicts: List[ScreeningVerdict]
    metrics: ModelMetrics
