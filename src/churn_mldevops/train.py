from __future__ import annotations

import json

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from churn_mldevops.config import ARTIFACTS_DIR, DATA_PATH, MODEL_PATH
from churn_mldevops.pipeline import load_and_prepare_data


def train_and_save() -> None:
    prepared = load_and_prepare_data(str(DATA_PATH))

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=60, random_state=0),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    scores = {}
    fitted_models = {}
    reports = {}
    for name, model in models.items():
        model.fit(prepared.X_train, prepared.y_train)
        pred = model.predict(prepared.X_test)
        score = accuracy_score(prepared.y_test, pred)
        scores[name] = float(score)
        fitted_models[name] = model
        reports[name] = classification_report(prepared.y_test, pred, output_dict=True)
        print(f"{name} accuracy: {score:.4f}")

    best_model_name = max(scores, key=scores.get)
    best_model = fitted_models[best_model_name]
    best_score = scores[best_model_name]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "accuracy": best_score,
        "encoders": prepared.encoders,
        "feature_columns": prepared.X_train.columns.tolist(),
        "train_reference_stats": prepared.train_reference_stats,
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "reports": reports}, f, indent=2)

    print(f"Best model: {best_model_name} ({best_score:.4f})")
    print(f"Saved artifact: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()

