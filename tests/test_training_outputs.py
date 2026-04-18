from __future__ import annotations

import json
import os
from pathlib import Path

import joblib


def test_manifest_and_metrics_written() -> None:
    art = Path(os.environ["ARTIFACTS_DIR"])
    assert (art / "manifest.json").exists()
    assert (art / "metrics.json").exists()
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["primary_metric"] == "f1"
    assert "data_sha256" in manifest
    metrics = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
    assert "per_model" in metrics
    assert len(metrics["per_model"]) >= 1


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
