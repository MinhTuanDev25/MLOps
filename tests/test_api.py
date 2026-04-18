from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from app import app

VALID_PAYLOAD = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 3,
    "Balance": 120_000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50_000.0,
}


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_name" in body
    assert body["manifest"] is not None
    assert body["manifest"]["selection"]["primary_metric"] == "f1"
    assert "metrics" in body


def test_predict_happy_path() -> None:
    client = TestClient(app)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability_exited"] <= 1.0
    assert "drift_check" in data
    assert "drift_flag" in data["drift_check"]
    assert "population_psi_vs_train_log" in data["drift_check"]


def test_predict_validation_422() -> None:
    client = TestClient(app)
    bad = {**VALID_PAYLOAD, "CreditScore": 100}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_geography_422() -> None:
    client = TestClient(app)
    bad = {**VALID_PAYLOAD, "Geography": "Italy"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_artifact_on_disk_matches_api() -> None:
    model_path = Path(os.environ["MODEL_PATH"])
    artifact = joblib.load(model_path)
    assert "model" in artifact
    assert "manifest" in artifact
    assert "reference_histograms" in artifact
