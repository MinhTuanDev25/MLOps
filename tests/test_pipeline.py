from __future__ import annotations

import os

import pytest

from churn_mldevops.pipeline import (
    load_and_prepare_data,
    prepare_single_record,
)


def test_load_and_prepare_data_shape() -> None:
    data_path = os.environ["DATA_PATH"]
    prepared = load_and_prepare_data(data_path)
    assert len(prepared.X_train) > 0
    assert len(prepared.X_test) > 0
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1]
    assert "Age" in prepared.reference_histograms
    assert "proportions" in prepared.reference_histograms["Age"]


def test_prepare_single_record_unknown_category() -> None:
    data_path = os.environ["DATA_PATH"]
    prepared = load_and_prepare_data(data_path)
    encoders = prepared.encoders
    bad_encoders = {
        "Geography": {"France": 0},
        "Gender": encoders["Gender"],
    }
    payload = {
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 3,
        "Balance": 0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 1000,
    }
    with pytest.raises(ValueError, match="Unknown categories"):
        prepare_single_record(payload, bad_encoders)


def test_stratified_target_both_classes_in_test() -> None:
    data_path = os.environ["DATA_PATH"]
    prepared = load_and_prepare_data(data_path)
    assert prepared.y_test.nunique() == 2
