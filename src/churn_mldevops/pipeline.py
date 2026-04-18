from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    encoders: Dict[str, Dict[str, int]]
    train_reference_stats: Dict[str, float]
    reference_histograms: Dict[str, Dict[str, Any]]


def _fit_category_mapping(series: pd.Series) -> Dict[str, int]:
    unique_values = sorted(series.astype(str).unique().tolist())
    return {value: idx for idx, value in enumerate(unique_values)}


def _apply_mapping(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    mapped = series.astype(str).map(mapping)
    if mapped.isnull().any():
        unknown = series[mapped.isnull()].astype(str).unique().tolist()
        raise ValueError(f"Unknown categories detected: {unknown}")
    return mapped.astype(int)


def histogram_props(series: pd.Series, bins: int = 10) -> Dict[str, Any]:
    values = series.astype(float).to_numpy()
    counts, bin_edges = np.histogram(values, bins=bins)
    total = counts.sum() or 1
    proportions = (counts / total).astype(float).tolist()
    return {
        "bin_edges": bin_edges.astype(float).tolist(),
        "proportions": proportions,
    }


def load_and_prepare_data(csv_path: str) -> PreparedData:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=DROP_COLUMNS)

    # Keep close to your notebook behavior.
    df["Balance"] = df["Balance"].astype(int)
    df["EstimatedSalary"] = df["EstimatedSalary"].astype(int)

    encoders = {
        "Geography": _fit_category_mapping(df["Geography"]),
        "Gender": _fit_category_mapping(df["Gender"]),
    }
    df["Geography"] = _apply_mapping(df["Geography"], encoders["Geography"])
    df["Gender"] = _apply_mapping(df["Gender"], encoders["Gender"])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=65, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    no_class = train_df[train_df[TARGET_COLUMN] == 0]
    yes_class = train_df[train_df[TARGET_COLUMN] == 1]
    no_class_downsampled = resample(
        no_class, replace=False, n_samples=len(yes_class), random_state=65
    )
    downsampled_train = pd.concat([yes_class, no_class_downsampled], axis=0)

    X_train_balanced = downsampled_train.drop(columns=[TARGET_COLUMN])
    y_train_balanced = downsampled_train[TARGET_COLUMN]

    train_reference_stats = {
        "age_mean": float(X_train_balanced["Age"].mean()),
        "balance_mean": float(X_train_balanced["Balance"].mean()),
    }

    reference_histograms = {
        "Age": histogram_props(X_train_balanced["Age"], bins=10),
        "Balance": histogram_props(X_train_balanced["Balance"].astype(float), bins=10),
    }

    return PreparedData(
        X_train=X_train_balanced,
        X_test=X_test,
        y_train=y_train_balanced,
        y_test=y_test,
        encoders=encoders,
        train_reference_stats=train_reference_stats,
        reference_histograms=reference_histograms,
    )


def prepare_single_record(
    payload: Dict[str, float | int | str], encoders: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    transformed = payload.copy()
    transformed["Geography"] = _apply_mapping(
        pd.Series([payload["Geography"]]), encoders["Geography"]
    ).iloc[0]
    transformed["Gender"] = _apply_mapping(
        pd.Series([payload["Gender"]]), encoders["Gender"]
    ).iloc[0]
    transformed["Balance"] = int(transformed["Balance"])
    transformed["EstimatedSalary"] = int(transformed["EstimatedSalary"])
    return pd.DataFrame([transformed])

