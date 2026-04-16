from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def append_prediction_log(
    log_path: Path, payload: dict, prediction: int, probability: float
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
        "prediction": int(prediction),
        "probability_exited": float(probability),
    }
    df_new = pd.DataFrame([row])
    if log_path.exists():
        df_existing = pd.read_csv(log_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(log_path, index=False)


def detect_basic_drift(latest_payload: dict, train_reference_stats: dict) -> dict:
    age_delta = abs(float(latest_payload["Age"]) - float(train_reference_stats["age_mean"]))
    balance_delta = abs(
        float(latest_payload["Balance"]) - float(train_reference_stats["balance_mean"])
    )
    return {
        "age_mean_delta": age_delta,
        "balance_mean_delta": balance_delta,
        "drift_flag": bool(age_delta > 15 or balance_delta > 80000),
    }

