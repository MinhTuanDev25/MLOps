from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_mldevops.config import MODEL_PATH, PREDICTION_LOG_PATH
from churn_mldevops.monitoring import append_prediction_log, build_drift_report
from churn_mldevops.pipeline import prepare_single_record


app = FastAPI(title="Churn Prediction API", version="1.0.0")


class ChurnRequest(BaseModel):
    CreditScore: int = Field(ge=300, le=900)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=10)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0)


def _load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifact not found. Train first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ChurnRequest) -> dict:
    artifact = _load_artifact()
    model = artifact["model"]
    encoders = artifact["encoders"]
    feature_columns = artifact["feature_columns"]
    train_reference_stats = artifact["train_reference_stats"]

    payload = request.model_dump()
    try:
        X = prepare_single_record(payload, encoders)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X = X[feature_columns]
    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    else:
        score = float(model.decision_function(X)[0])
        proba = 1.0 / (1.0 + np.exp(-score))

    append_prediction_log(PREDICTION_LOG_PATH, payload, pred, proba)
    reference_histograms = artifact.get("reference_histograms")
    drift = build_drift_report(
        payload,
        train_reference_stats,
        reference_histograms,
        PREDICTION_LOG_PATH,
    )

    return {
        "model_name": artifact["model_name"],
        "model_version": (artifact.get("manifest") or {}).get("created_at_utc"),
        "prediction": pred,
        "probability_exited": round(proba, 6),
        "drift_check": drift,
    }


@app.get("/model-info")
def model_info() -> dict:
    artifact = _load_artifact()
    manifest = artifact.get("manifest")
    metrics = artifact.get("metrics")
    return {
        "model_name": artifact["model_name"],
        "metrics": metrics,
        "manifest": manifest,
    }

