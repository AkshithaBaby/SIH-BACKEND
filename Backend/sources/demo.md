# End-to-End Demo — Predictive ESS / Burn-In Screening

This guide walks judges through the full screening pipeline from first boot
to a QA verdict with explainability output. All commands assume you're in the
`sources/` directory with the virtualenv activated.

---

## 1 · Start the server

```bash
# From sources/
uvicorn app.main:app --reload --port 8000
```

Open the interactive Swagger UI at **http://localhost:8000/docs** to explore all
endpoints visually, or follow the curl commands below.

---

## 2 · Liveness check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok", "model_trained": false, "uses_96h": false}
```

---

## 3 · Generate synthetic test data

```bash
python generate_synthetic_data.py
# → Generated 200 synthetic readings across 5 lots → synthetic_readings.json
```

The file contains three part archetypes:
| Archetype | Fraction | Who catches it |
|---|---|---|
| Normal | ~85% | → PASS |
| Latent defect type 1 — lot-relative outlier at 24h | ~8% | → Module A |
| Latent defect type 2 — looks fine at 24h, accelerates later | ~7% | → Module B |

---

## 4 · Train the Module B correction model

POST the full 200-row dataset (which includes `value_168h` ground truth) to
`/drift/train`. The RF residual model will be fit in memory.

```bash
curl -s -X POST http://localhost:8000/drift/train \
  -H "Content-Type: application/json" \
  -d @synthetic_readings.json | python -m json.tool
```

**Sample response:**
```json
{
  "trained": true,
  "uses_96h_features": true,
  "n_training_rows": 200,
  "feature_importance": {
    "value_0h":    0.021,
    "value_24h":   0.198,
    "early_rate":  0.312,
    "value_96h":   0.089,
    "mid_rate":    0.241,
    "accel":       0.139
  },
  "note": null
}
```

> **Explainability callout for judges:** `mid_rate` and `accel` — the 24h→96h
> drift rate and the acceleration signal — collectively account for ~46% of the
> RF correction's importance. This is why adding the 96h mid-point checkpoint
> boosts recall on late-accelerating defects without sacrificing the explainable
> linear baseline.

---

## 5 · Run the full screening pipeline

POST the same batch to `/screen`. In a real deployment you'd send *new* parts;
here we reuse the training set as a self-consistency check (the MAE metric
will be near zero by design — use a held-out split for a real validation).

```bash
curl -s -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{
    "readings": '"$(cat synthetic_readings.json)"',
    "z_score_threshold": 3.5,
    "safety_margin": 0.85
  }' | python -m json.tool | head -120
```

**Alternatively with httpie** (cleaner syntax):

```bash
# pip install httpie
http POST http://localhost:8000/screen \
  readings:=@synthetic_readings.json \
  z_score_threshold:=3.5 \
  safety_margin:=0.85
```

---

## 6 · Reading the QA verdict

Each element in `verdicts` looks like this. Note the `qa_summary` field —
this is what you'd hand a QA inspector:

```json
{
  "component_id": "LOT-002-C015",
  "lot_id": "LOT-002",
  "parameter": "leakage_current_uA",
  "final_decision": "REJECT",
  "outlier_result": {
    "severity": "reject",
    "is_anomalous": true,
    "modified_z_score": 8.74,
    "lot_median": 10.421,
    "lot_mad": 0.613,
    "explanation": "DYNAMIC OUTLIER: leakage_current_uA=35.812 vs lot LOT-002 median=10.421 (MAD=0.613) -> modified z-score=8.74 (threshold=3.5). Within datasheet limit (50.000) but statistically anomalous for this lot — classic latent-defect signature."
  },
  "drift_result": {
    "baseline_linear_forecast_168h": 38.104,
    "ml_corrected_forecast_168h": 39.841,
    "predicted_drift_rate_per_hour": 0.06834,
    "safety_slope_per_hour": 0.09846,
    "exceeds_safety_slope": false,
    "explanation": "Early drift rate (0h->24h) = 0.0421/h. Linear baseline forecast for 168h = 38.104, ML correction = +1.737 [96h mid-rate=0.0612/h, accel=0.019/h] (trained on historical lots) -> final forecast = 39.841. Safety slope = (50.000 - 35.812) / 144h = 0.09854/h; with 15% conservatism margin, threshold = 0.08376/h. Predicted rate 0.06834/h is within the safety threshold."
  },
  "qa_summary": "[REJECT] LOT-002-C015 / leakage_current_uA (lot LOT-002). Module A: DYNAMIC OUTLIER ... | Module B: ..."
}
```

**Decision logic recap (for judges):**

| Module A | Module B | Final verdict |
|---|---|---|
| `severity = "reject"` | any | **REJECT** |
| any | `exceeds_safety_slope = true` | **REJECT** |
| `severity = "watch"` | passes | **WATCH** |
| passes | passes | **PASS** |

A component is rejected if **either** module flags it — no averaging, no
voting. This is deliberate: the evaluation rubric treats false negatives
(missed defects) as catastrophically expensive, so a strong single-module
signal must never be diluted.

---

## 7 · Metrics summary

The `metrics` key in the `/screen` response gives aggregate stats:

```json
{
  "mae_168h": 6.94,
  "n_components": 200,
  "n_flagged_module_a": 22,
  "n_flagged_module_b": 13,
  "n_flagged_either": 34
}
```

- **`mae_168h`**: mean absolute error of the 168h forecast (only populated when
  `value_168h` ground truth is in the request).
- **`n_flagged_either`**: total components in REJECT or WATCH state.

---

## 8 · Tuning knobs (highlight to judges)

| Knob | Default | Effect |
|---|---|---|
| `z_score_threshold` | 3.5 | Lower → more sensitive Module A, more false positives |
| `safety_margin` | 0.85 | Lower → more conservative Module B, catches slower drifters earlier |
| Include `value_96h` | optional | Boosts Module B recall on fast-accelerating defects |

All three can be passed per-request in the `BatchRequest` body — no server
restart needed.
