import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("DATA_PATH", str(ROOT_DIR / "Churn_Modelling.csv")))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT_DIR / "artifacts")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "best_model.pkl")))
_DEFAULT_SQLITE = f"sqlite:///{(ARTIFACTS_DIR / 'churn.db').resolve().as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_SQLITE)

