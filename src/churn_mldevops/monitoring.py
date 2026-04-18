from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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


def psi(expected_props: list[float], actual_props: list[float], eps: float = 1e-6) -> float:
    e = np.asarray(expected_props, dtype=float) + eps
    a = np.asarray(actual_props, dtype=float) + eps
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _proportions_with_bins(values: np.ndarray, bin_edges: list[float]) -> list[float]:
    counts, _ = np.histogram(values, bins=np.asarray(bin_edges, dtype=float))
    total = counts.sum() or 1
    return (counts / total).astype(float).tolist()


def population_psi_vs_train(
    log_path: Path,
    column: str,
    reference_histogram: dict[str, Any],
    last_n: int = 200,
    min_rows: int = 30,
) -> float | None:
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    if column not in df.columns or len(df) < min_rows:
        return None
    window = df[column].astype(float).tail(last_n).to_numpy()
    ref_edges = reference_histogram["bin_edges"]
    ref_props = reference_histogram["proportions"]
    actual_props = _proportions_with_bins(window, ref_edges)
    if len(actual_props) != len(ref_props):
        return None
    return psi(ref_props, actual_props)


def build_drift_report(
    latest_payload: dict,
    train_reference_stats: dict,
    reference_histograms: dict[str, dict[str, Any]] | None,
    log_path: Path,
    psi_threshold: float = 0.25,
    log_window: int = 200,
) -> dict[str, Any]:
    basic = detect_basic_drift(latest_payload, train_reference_stats)
    combined_flag = basic["drift_flag"]
    psi_block: dict[str, float | None] = {}

    if reference_histograms:
        for col in ("Age", "Balance"):
            hist = reference_histograms.get(col)
            if not hist:
                continue
            val = population_psi_vs_train(log_path, col, hist, last_n=log_window)
            psi_block[col] = val
            if val is not None and val > psi_threshold:
                combined_flag = True

    return {
        "basic_rules": basic,
        "population_psi_vs_train_log": psi_block,
        "psi_threshold": psi_threshold,
        "drift_flag": combined_flag,
    }
