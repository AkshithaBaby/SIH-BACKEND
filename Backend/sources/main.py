"""
FastAPI backend for Predictive ESS / Burn-In Screening.

Run:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /outlier/analyze   -> Module A only
    POST /drift/predict     -> Module B only (fit on the same batch's
                                 known-168h rows if any, else baseline-only)
    POST /screen            -> Full pipeline -> QA verdicts + metrics
    POST /drift/train       -> Explicitly train the drift correction model
                                 on historical data with known 168h values,
                                 kept in memory for subsequent /screen calls
    GET  /health            -> liveness check
"""

from fastapi import FastAPI
from typing import List

from app.schemas import (
    BatchRequest,
    ParameterReading,
    OutlierFlag,
    DriftPrediction,
    ScreeningVerdict,
    ModelMetrics,
)
from app.module_a_outlier import detect_outliers
from app.module_b_drift import DriftModel
from app.pipeline import run_screening

app = FastAPI(
    title="Predictive ESS Screening Backend",
    description="Dynamic outlier detection + drift prediction for burn-in latent defect screening.",
    version="0.1.0",
)

# In-memory model instance. For the hackathon this is fine; for production
# swap for a persisted model (joblib dump) loaded per-request or per-worker.
_drift_model = DriftModel()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/outlier/analyze", response_model=List[OutlierFlag])
def analyze_outliers(req: BatchRequest):
    return detect_outliers(req.readings, checkpoint="value_24h", z_threshold=req.z_score_threshold)


@app.post("/drift/train")
def train_drift_model(readings: List[ParameterReading]):
    """
    Train (or retrain) the Module B correction model on historical
    readings that include ground-truth value_168h. Kept in memory for
    subsequent /drift/predict and /screen calls in this process.
    """
    global _drift_model
    _drift_model = DriftModel().fit(readings)
    return {
        "trained": _drift_model.is_fit,
        "n_training_rows": len(readings),
        "feature_importance": _drift_model.feature_importance_report(),
        "note": None if _drift_model.is_fit else "Fewer than 10 labeled rows — using explainable linear baseline only.",
    }


@app.post("/drift/predict", response_model=List[DriftPrediction])
def predict_drift(req: BatchRequest):
    return _drift_model.predict_batch(req.readings, safety_margin=req.safety_margin)


@app.post("/screen")
def screen(req: BatchRequest):
    verdicts, metrics = run_screening(
        req.readings,
        drift_model=_drift_model,
        z_threshold=req.z_score_threshold,
        safety_margin=req.safety_margin,
    )
    return {"verdicts": verdicts, "metrics": metrics}
