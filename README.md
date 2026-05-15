# Churn MLOps

Personal project for learning **MLOps**: train a bank **churn** classifier on `Churn_Modelling.csv`, serve it with **FastAPI**, persist predictions and training history in **PostgreSQL** (or **SQLite** locally), monitor with **Prometheus + Grafana**, and ship via **Docker** and **GitHub Actions**.

```mermaid
flowchart LR
  CSV[Churn_Modelling.csv] --> Train[train.py]
  Train --> Art[(artifacts/best_model.pkl)]
  Art --> API[FastAPI :8000]
  Client[Client] --> API
  API --> DB[(PostgreSQL / SQLite)]
  API --> PM[/metrics]
  Prom[Prometheus] --> PM
  Graf[Grafana] --> Prom
```

---

## What's here

| Path | Description |
|------|-------------|
| `train.py` | Entry point for training |
| `app.py` | FastAPI inference API |
| `src/churn_mldevops/` | Config, pipeline, training, monitoring, ORM |
| `tests/` | pytest (train into temp dirs in CI) |
| `scripts/responsible_ai_report.py` | Offline fairness / importance report |
| `observability/` | Prometheus config + Grafana dashboards |
| `docker-compose.yml` | Postgres, API, Prometheus, Grafana |
| `.github/workflows/ci.yml` | Test → Docker build → GHCR + Render hook |

**Stack:** Python 3.11 · scikit-learn · FastAPI · SQLAlchemy · Prometheus · Grafana · Docker

**Model selection:** compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, LinearSVC — pick best **F1** on test set (tie-break **ROC-AUC**, then **average precision**).

---

## Requirements

- Python **3.11+**
- `Churn_Modelling.csv` in the project root (or set `DATA_PATH`)

---

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python train.py
uvicorn app:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness |
| `POST /predict` | Churn prediction + drift check |
| `GET /model-info` | Manifest and metrics from served artifact |
| `GET /training-runs` | Latest runs from DB |
| `GET /metrics` | Prometheus scrape (text) |
| `GET /metrics/json` | Same registry as JSON (for demos) |
| `GET /docs` | Swagger UI |

**Training outputs:** `artifacts/best_model.pkl` (model + encoders + reference stats) and table `training_runs` in the DB.

---

## Docker (full stack)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://127.0.0.1:8000 |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000 (`admin` / `admin`) |

The API image runs `train.py` at **build** time. With Compose, set `DATABASE_URL` to Postgres; after the stack is up, sync training history if needed:

```bash
docker compose exec churn-api python train.py
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `./Churn_Modelling.csv` | Training CSV |
| `ARTIFACTS_DIR` | `./artifacts` | Models and local SQLite path |
| `MODEL_PATH` | `$ARTIFACTS_DIR/best_model.pkl` | Inference artifact |
| `DATABASE_URL` | `sqlite:///…/artifacts/churn.db` | SQLAlchemy URL; Compose uses Postgres |

---

## Monitoring & drift

- **Rule drift:** age / balance vs training means on the balanced train set.
- **PSI:** Age and Balance vs reference histograms from recent `predictions` rows (min ~30); default threshold **0.25**.
- **Metrics:** `churn_predictions_total`, `churn_drift_events_total`, plus FastAPI instrumentator HTTP metrics.

```bash
python scripts/responsible_ai_report.py   # writes artifacts/responsible_ai_report.json
```

---

## Tests & CI/CD

```bash
pytest -q tests
```

On push: **pytest** → **Docker build** → on `main`: push to **GHCR** and trigger **Render** deploy hook (`RENDER_DEPLOY_HOOK_URL` secret). See `render.yaml` for the Render web service (`/health`).

---

## Author

**Minh Tuan**
