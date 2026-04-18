from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from churn_mldevops.config import ARTIFACTS_DIR, DATA_PATH, MODEL_PATH
from churn_mldevops.pipeline import load_and_prepare_data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    if os.getenv("GITHUB_SHA"):
        return os.getenv("GITHUB_SHA", "")[:40]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()[:40]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _scores_for_model(model: Any, X_test: Any, y_test: Any) -> dict[str, float]:
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
    try:
        roc = float(roc_auc_score(y_test, scores))
    except ValueError:
        roc = float("nan")
    try:
        ap = float(average_precision_score(y_test, scores))
    except ValueError:
        ap = float("nan")
    return {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "average_precision": ap,
    }


def _pick_best_model(per_model: dict[str, dict[str, float]]) -> str:
    def sort_key(name: str) -> tuple[float, float, float]:
        m = per_model[name]
        f1 = m.get("f1", 0.0)
        roc = m.get("roc_auc", float("-inf"))
        ap = m.get("average_precision", float("-inf"))
        if roc != roc:  # NaN
            roc = -1.0
        if ap != ap:
            ap = -1.0
        return (f1, roc, ap)

    return max(per_model.keys(), key=sort_key)


def train_and_save() -> None:
    prepared = load_and_prepare_data(str(DATA_PATH))

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=3000),
        "LinearSVC": LinearSVC(random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=60, random_state=0),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    per_model_scores: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(prepared.X_train, prepared.y_train)
        metrics = _scores_for_model(model, prepared.X_test, prepared.y_test)
        per_model_scores[name] = metrics
        fitted_models[name] = model
        pred = model.predict(prepared.X_test)
        reports[name] = classification_report(prepared.y_test, pred, output_dict=True)
        print(
            f"{name}: f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} ap={metrics['average_precision']:.4f}"
        )

    best_model_name = _pick_best_model(per_model_scores)
    best_model = fitted_models[best_model_name]
    best_metrics = per_model_scores[best_model_name]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data_sha = _sha256_file(Path(DATA_PATH)) if Path(DATA_PATH).exists() else None
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA_PATH),
        "data_sha256": data_sha,
        "git_commit": _git_commit(),
        "sklearn_version": sklearn.__version__,
        "selection": {
            "primary_metric": "f1",
            "tie_breakers": ["roc_auc", "average_precision"],
            "best_model": best_model_name,
        },
        "best_test_metrics": best_metrics,
        "all_test_metrics": per_model_scores,
    }
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "metrics": best_metrics,
        "encoders": prepared.encoders,
        "feature_columns": prepared.X_train.columns.tolist(),
        "train_reference_stats": prepared.train_reference_stats,
        "reference_histograms": prepared.reference_histograms,
        "manifest": manifest,
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_model": per_model_scores,
                "reports": reports,
                "selection": manifest["selection"],
            },
            f,
            indent=2,
        )

    print(f"Best model (by F1, then ROC-AUC, then AP): {best_model_name}")
    print(f"Best metrics: {best_metrics}")
    print(f"Saved artifact: {MODEL_PATH}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    train_and_save()
