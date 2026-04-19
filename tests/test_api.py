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


def test_prometheus_metrics_endpoint() -> None:
    client = TestClient(app)
    client.post("/predict", json=VALID_PAYLOAD)
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "http_requests_total" in body or "http_request_duration" in body
    assert "churn_predictions_total" in body


def test_metrics_json_endpoint() -> None:
    client = TestClient(app)
    client.post("/predict", json=VALID_PAYLOAD)
    response = client.get("/metrics/json")
    assert response.status_code == 200
    data = response.json()
    assert data.get("format") == "prometheus_registry_json"
    assert "samples" in data
    names = {s["name"] for s in data["samples"]}
    assert "churn_predictions_total" in names


def test_training_runs_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/training-runs?limit=3")
    assert response.status_code == 200
    body = response.json()
    assert "runs" in body
    assert len(body["runs"]) >= 1
    assert body["runs"][0]["manifest"]["selection"]["primary_metric"] == "f1"


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
