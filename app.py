from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_mldevops.config import MODEL_PATH
from churn_mldevops.database import init_db, session_scope
from churn_mldevops.monitoring import append_prediction_row, build_drift_report
from churn_mldevops.orm_models import TrainingRun
from churn_mldevops.pipeline import prepare_single_record


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

churn_predictions_total = Counter(
    "churn_predictions_total",
    "Churn predictions by model and predicted class",
    ["model_name", "exited"],
)
churn_drift_events_total = Counter(
    "churn_drift_events_total",
    "Requests where drift_flag evaluated true (rules or PSI)",
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


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

    reference_histograms = artifact.get("reference_histograms")

    with session_scope() as session:
        append_prediction_row(
            session,
            payload,
            pred,
            proba,
            str(artifact["model_name"]),
        )
        drift = build_drift_report(
            payload,
            train_reference_stats,
            reference_histograms,
            session,
        )

    churn_predictions_total.labels(
        model_name=str(artifact["model_name"]), exited=str(pred)
    ).inc()
    if drift.get("drift_flag"):
        churn_drift_events_total.inc()

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


@app.get("/training-runs")
def training_runs(limit: int = 10) -> dict:
    """Latest training runs persisted in DB (manifest includes per-model metrics)."""
    lim = max(1, min(limit, 50))
    with session_scope() as session:
        rows = session.scalars(
            select(TrainingRun).order_by(TrainingRun.id.desc()).limit(lim)
        ).all()
    return {
        "runs": [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "best_model": r.manifest.get("selection", {}).get("best_model"),
                "manifest": r.manifest,
                "classification_reports": r.classification_reports,
            }
            for r in rows
        ]
    }
