"""Train into a temp artifact dir before importing the API (config reads env at import)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_TEST_ART = tempfile.mkdtemp(prefix="churn_pytest_artifacts_")
os.environ["ARTIFACTS_DIR"] = _TEST_ART
os.environ["MODEL_PATH"] = str(Path(_TEST_ART) / "best_model.pkl")
os.environ["DATA_PATH"] = str(_ROOT / "Churn_Modelling.csv")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TEST_ART, 'churn.db').resolve().as_posix()}"

from churn_mldevops.train import train_and_save  # noqa: E402

train_and_save()
