from __future__ import annotations

import os
from pathlib import Path

import joblib
from sqlalchemy import select

from churn_mldevops.database import session_scope
from churn_mldevops.orm_models import TrainingRun


def test_training_run_in_database() -> None:
    with session_scope() as session:
        row = session.scalars(select(TrainingRun).order_by(TrainingRun.id.desc())).first()
    assert row is not None
    assert row.manifest["selection"]["primary_metric"] == "f1"
    assert "data_sha256" in row.manifest
    assert isinstance(row.classification_reports, dict)
    assert len(row.classification_reports) >= 1


def test_model_load_and_predict() -> None:
    from churn_mldevops.pipeline import prepare_single_record

    model_path = Path(os.environ["MODEL_PATH"])
    artifact = joblib.load(model_path)
    model = artifact["model"]
    encoders = artifact["encoders"]
    cols = artifact["feature_columns"]
    payload = {
        "CreditScore": 700,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50_000,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 40_000,
    }
    X = prepare_single_record(payload, encoders)[cols]
    pred = model.predict(X)
    assert pred.shape == (1,)
    assert int(pred[0]) in (0, 1)
