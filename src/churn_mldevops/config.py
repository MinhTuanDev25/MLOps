import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("DATA_PATH", str(ROOT_DIR / "Churn_Modelling.csv")))
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "best_model.pkl")))
PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", str(ARTIFACTS_DIR / "prediction_log.csv"))
)

