"""Offline Responsible AI report: slice metrics + permutation importance.

Run from repo root: python scripts/responsible_ai_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_mldevops.config import ARTIFACTS_DIR, DATA_PATH, MODEL_PATH  # noqa: E402
from churn_mldevops.pipeline import load_and_prepare_data  # noqa: E402


def _slice_metrics(
    group_labels: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in sorted(np.unique(group_labels)):
        mask = group_labels == g
        out[str(g)] = {
            "n": int(mask.sum()),
            "prevalence_true": float(y_true[mask].mean()) if mask.any() else 0.0,
            "positive_rate_pred": float(y_pred[mask].mean()) if mask.any() else 0.0,
            "f1": float(f1_score(y_true[mask], y_pred[mask], zero_division=0)),
        }
    return out


def main() -> None:
    prepared = load_and_prepare_data(str(DATA_PATH))
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        feature_columns = list(artifact["feature_columns"])
    else:
        model = LogisticRegression(max_iter=3000)
        model.fit(prepared.X_train, prepared.y_train)
        feature_columns = prepared.X_train.columns.tolist()

    X_test = prepared.X_test[feature_columns]
    y_test = prepared.y_test.to_numpy()
    y_pred = model.predict(X_test)

    inv_geo = {v: k for k, v in prepared.encoders["Geography"].items()}
    inv_gen = {v: k for k, v in prepared.encoders["Gender"].items()}
    geo_labels = prepared.X_test["Geography"].map(inv_geo).astype(str).to_numpy()
    gen_labels = prepared.X_test["Gender"].map(inv_gen).astype(str).to_numpy()

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=0,
        n_jobs=1,
    )
    importance = sorted(
        zip(feature_columns, perm.importances_mean),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:12]

    report = {
        "limitations": [
            "Dataset is static and may not reflect current customer behavior.",
            "Churn labels are historical; the model is associative, not causal.",
            "Protected attributes appear as inputs; deployment should follow policy and law.",
            "Do not use raw scores alone for high-stakes decisions without human review.",
        ],
        "overall_test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "by_geography": _slice_metrics(geo_labels, y_test, y_pred),
        "by_gender": _slice_metrics(gen_labels, y_test, y_pred),
        "permutation_importance_top": [
            {"feature": name, "mean_delta_accuracy": float(mean)} for name, mean in importance
        ],
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "responsible_ai_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
