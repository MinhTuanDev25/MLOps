from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from churn_mldevops.orm_models import Prediction


def append_prediction_row(
    session: Session,
    payload: dict,
    prediction: int,
    probability: float,
    model_name: str,
) -> None:
    row = Prediction(
        credit_score=int(payload["CreditScore"]),
        geography=str(payload["Geography"]),
        gender=str(payload["Gender"]),
        age=int(payload["Age"]),
        tenure=int(payload["Tenure"]),
        balance=float(payload["Balance"]),
        num_of_products=int(payload["NumOfProducts"]),
        has_cr_card=int(payload["HasCrCard"]),
        is_active_member=int(payload["IsActiveMember"]),
        estimated_salary=float(payload["EstimatedSalary"]),
        prediction=int(prediction),
        probability_exited=float(probability),
        model_name=model_name,
    )
    session.add(row)
    session.flush()


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


_PSI_COLUMN = {"Age": Prediction.age, "Balance": Prediction.balance}


def population_psi_vs_train_db(
    session: Session,
    column: str,
    reference_histogram: dict[str, Any],
    last_n: int = 200,
    min_rows: int = 30,
) -> float | None:
    col = _PSI_COLUMN.get(column)
    if col is None:
        return None
    stmt = select(col).order_by(desc(Prediction.id)).limit(last_n)
    rows = session.scalars(stmt).all()
    if len(rows) < min_rows:
        return None
    window = np.asarray(rows, dtype=float)
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
    session: Session,
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
            val = population_psi_vs_train_db(session, col, hist, last_n=log_window)
            psi_block[col] = val
            if val is not None and val > psi_threshold:
                combined_flag = True

    return {
        "basic_rules": basic,
        "population_psi_vs_train_log": psi_block,
        "psi_threshold": psi_threshold,
        "drift_flag": combined_flag,
    }
