"""
Generates synthetic burn-in parametric data for testing the backend before
real ESS data is available. Simulates:
  - Normal parts: mild, roughly linear drift.
  - Latent-defect parts: pass 24h checks but accelerate later (the exact
    failure mode Module A/B are designed to catch), OR are a subtle
    lot-relative outlier from the start while staying under datasheet max.

Output: synthetic_readings.json — a list matching the ParameterReading schema.
"""

import json
import numpy as np

rng = np.random.default_rng(7)

PARAMETER = "leakage_current_uA"
DATASHEET_MAX = 50.0
N_LOTS = 5
PARTS_PER_LOT = 40

records = []

for lot_num in range(N_LOTS):
    lot_id = f"LOT-{lot_num+1:03d}"
    lot_baseline = rng.uniform(8, 14)  # this lot's "normal" leakage baseline

    for part_num in range(PARTS_PER_LOT):
        component_id = f"{lot_id}-C{part_num+1:03d}"

        roll = rng.random()
        if roll < 0.85:
            # Normal part: small noise, gentle linear drift
            v0 = lot_baseline + rng.normal(0, 0.8)
            rate = rng.normal(0.02, 0.01)  # uA/hour
            v24 = v0 + rate * 24 + rng.normal(0, 0.3)
            v96 = v0 + rate * 96 + rng.normal(0, 0.5)
            v168 = v0 + rate * 168 + rng.normal(0, 0.6)
        elif roll < 0.93:
            # Latent defect type 1: within datasheet limit at 24h, but a
            # clear lot-relative outlier already (Module A should catch it)
            v0 = lot_baseline + rng.normal(0, 0.8)
            v24 = lot_baseline * rng.uniform(3.0, 4.2)  # e.g. 10uA lot -> 35-40uA part
            rate = rng.normal(0.05, 0.02)
            v96 = v24 + rate * 72 + rng.normal(0, 1)
            v168 = v24 + rate * 144 + rng.normal(0, 1.5)
        else:
            # Latent defect type 2: looks fine at 0h/24h, accelerates later
            # (Module B should catch this via drift-rate forecasting)
            v0 = lot_baseline + rng.normal(0, 0.8)
            v24 = v0 + rng.normal(0.5, 0.3)
            accel_rate = rng.uniform(0.25, 0.4)  # much steeper post-24h
            v96 = v24 + accel_rate * 72 + rng.normal(0, 1)
            v168 = v24 + accel_rate * 144 + rng.normal(0, 1.5)

        v168 = min(v168, DATASHEET_MAX * 1.15)  # allow some to genuinely exceed spec

        records.append(
            {
                "component_id": component_id,
                "lot_id": lot_id,
                "parameter": PARAMETER,
                "value_0h": round(float(v0), 3),
                "value_24h": round(float(v24), 3),
                "value_96h": round(float(v96), 3),
                "value_168h": round(float(v168), 3),
                "datasheet_max": DATASHEET_MAX,
                "datasheet_min": 0.0,
            }
        )

with open("synthetic_readings.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"Generated {len(records)} synthetic readings across {N_LOTS} lots -> synthetic_readings.json")
