# Churn MLDevOps

Đồ án / lab **MLOps**: dự đoán **churn** trên `Churn_Modelling.csv` — **train** (nhiều model sklearn, chọn theo F1) → **artifact** `best_model.pkl` + **lịch sử train trong DB** → **FastAPI** → **ghi predictions vào DB** + **drift (rule + PSI)** → **PostgreSQL** (Compose) hoặc **SQLite** (local mặc định) → **Docker** → **GitHub Actions** (pytest, build image, deploy hook).

---

## Kiến trúc tóm tắt

```mermaid
flowchart LR
  CSV[Churn_Modelling.csv] --> Train[train.py]
  Train --> Art[(artifacts)]
  Art --> API[FastAPI]
  Client[Client] --> API
  API --> DB[(PostgreSQL / SQLite)]
  API --> PM[/metrics]
  Prom[Prometheus] --> PM
  Graf[Grafana] --> Prom
```

---

## Yêu cầu

- Python **3.11+** (Dockerfile dùng 3.11)
- File dữ liệu `Churn_Modelling.csv` ở thư mục gốc project (hoặc chỉnh `DATA_PATH`)

---

## Cài đặt & chạy local

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python train.py
uvicorn app:app --reload --port 8000
```

- **OpenAPI / Swagger:** http://127.0.0.1:8000/docs  
- **Health:** `GET /health`  
- **Dự đoán:** `POST /predict`  
- **Metadata model:** `GET /model-info` (manifest + metric — lấy từ artifact đang serve)  
- **Lịch sử train (DB):** `GET /training-runs`  
- **Prometheus:** `GET /metrics` (chuẩn exposition text)

---

## Observability: Prometheus + Grafana

Stack kèm trong **`docker compose`**:

| Dịch vụ | URL | Ghi chú |
|---------|-----|---------|
| API | http://127.0.0.1:8000 | FastAPI |
| Prometheus | http://127.0.0.1:9090 | Scrape `churn-api:8000/metrics` (cấu hình `observability/prometheus.yml`) |
| Grafana | http://127.0.0.1:3000 | User / pass mặc định: **`admin` / `admin`** (đổi ngay khi demo xong) |

- **Datasource** Prometheus được provision sẵn (UID `prometheus`).
- **Dashboard** mẫu: *Churn API — Prometheus* (HTTP rate, prediction rate, drift flags, latency p95).
- Metric từ API:
  - `prometheus-fastapi-instrumentator`: latency + `http_requests_total`
  - Custom: `churn_predictions_total{model_name, exited}`, `churn_drift_events_total`

Chạy toàn stack:

```bash
docker compose up --build
```

**Lưu ý:** build image API chạy `train.py` — lần đầu có thể vài phút. Sau khi API lên, vài giây sau Prometheus mới có điểm scrape; Grafana có thể cần refresh dashboard.

---

## Docker

```bash
docker compose up --build
```

Services: **PostgreSQL** → **churn-api** → Prometheus → Grafana. Image API vẫn **`RUN python train.py`** lúc **build** (tạo `best_model.pkl` và SQLite `churn.db` trong layer image). Khi chạy Compose, API dùng **`DATABASE_URL` trỏ Postgres** — gợi ý sau khi stack lên, đồng bộ một lần artifact + bảng `training_runs`:

```bash
docker compose exec churn-api python train.py
```

(Nếu mount `./artifacts` từ máy và đã có `best_model.pkl`, có thể bỏ qua bước trên; khi đó bảng `training_runs` có thể trống cho đến khi bạn train lại với DB Postgres.)

---

## Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|--------|
| `DATA_PATH` | `./Churn_Modelling.csv` | CSV huấn luyện |
| `ARTIFACTS_DIR` | `./artifacts` | Thư mục chứa `best_model.pkl` và (local) `churn.db` nếu dùng SQLite |
| `MODEL_PATH` | `$ARTIFACTS_DIR/best_model.pkl` | Artifact inference (luôn là **file** — chuẩn ML) |
| `DATABASE_URL` | `sqlite:///…/artifacts/churn.db` | SQLAlchemy URL; Compose dùng `postgresql+psycopg2://churn:churn@db:5432/churn` |

---

## Huấn luyện & output

```bash
python train.py
```

Sinh:

- **`best_model.pkl`** — model sklearn + encoders + `feature_columns` + histogram/stats + `manifest` nhúng trong artifact (để serve không phụ thuộc DB).
- **Bảng `training_runs`** — một dòng mỗi lần train: JSON **manifest** (metadata, hash data, metric từng model…) và **classification_reports**.

Không còn ghi `manifest.json` / `metrics.json` riêng; tra cứu qua DB (`GET /training-runs`) hoặc artifact trong pickle.

**Chọn model:** ưu tiên **F1** trên tập test, tie-break **ROC-AUC** → **average precision**.

---

## Monitoring (API)

- **Rule drift:** lệch tuổi / số dư so với mean trên tập train đã cân bằng.
- **PSI:** so phân phối **Age** và **Balance** trên **cửa sổ các dòng gần nhất trong bảng `predictions`** (tối thiểu ~30 dòng) với histogram lúc train; ngưỡng PSI mặc định **0.25**.

Chi tiết: xem `src/churn_mldevops/monitoring.py` và doc A–Z.

---

## Responsible AI (offline)

```bash
python scripts/responsible_ai_report.py
```

Ghi `artifacts/responsible_ai_report.json`: slice theo Geography/Gender, permutation importance, mục **limitations**.

---

## Tests

```bash
pytest -q tests
```

`tests/conftest.py` train vào **thư mục tạm** (qua `ARTIFACTS_DIR` / `MODEL_PATH`) để CI không phụ thuộc `artifacts/` local.

---

## CI/CD

File `.github/workflows/ci.yml`:

1. Cài dependency → **pytest**
2. **Docker build** (sau khi test pass)
3. Trên nhánh `main`: push image lên **GHCR** + gọi **Render deploy hook** (cần secret `RENDER_DEPLOY_HOOK_URL`)

`render.yaml` mô tả service web Docker trên Render (health: `/health`). Trên cloud nên gắn **PostgreSQL** (Render Postgres) và set **`DATABASE_URL`** trong dashboard — SQLite trên ephemeral disk không bền giữa các lần deploy.

---

## Cấu trúc thư mục chính

| Đường dẫn | Nội dung |
|-----------|----------|
| `app.py` | FastAPI |
| `train.py` | Gọi `churn_mldevops.train.train_and_save` |
| `src/churn_mldevops/` | `config`, `pipeline`, `train`, `monitoring` |
| `tests/` | pytest |
| `scripts/responsible_ai_report.py` | Báo cáo RA |
| `docs/PROJECT_KNOWLEDGE_A_TO_Z.md` | Kiến thức & câu hỏi viva |
| `observability/` | `prometheus.yml`, Grafana provisioning + dashboard |
| `src/churn_mldevops/orm_models.py`, `database.py` | SQLAlchemy: `predictions`, `training_runs` |

---

## Thiết kế (tóm tắt rubric)

| Quyết định | Lý do ngắn |
|------------|------------|
| **FastAPI** | Schema/Pydantic, OpenAPI sẵn, hợp microservice scoring |
| **Model file (`joblib`)** | Đơn giản, tái lập; kèm `manifest` để truy vết |
| **Predictions trong DB** | Append-only SQL, query/filter; Prometheus cho dashboard thời gian thực |
| **Prometheus + Grafana** | Chuẩn quan sát SRE: scrape `/metrics`, visualize & alert rules sau này |

---

## License / môn học

Dự án phục vụ mục đích học tập (FPT / MLOps). Dữ liệu `Churn_Modelling` thường dùng trong các khóa ML công khai.
