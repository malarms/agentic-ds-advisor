
import pandas as pd
import pytest

from app.tools.model_training import train_and_compare_models


def test_classification_trains_two_models():
    df = pd.DataFrame({
        "age": list(range(20, 80)),
        "salary": list(range(30000, 90000, 1000)),
        "city": ["Pune", "Mumbai", "Delhi"] * 20,
        "purchased": ["yes", "no"] * 30,
    })

    result = train_and_compare_models(
        df,
        target_column="purchased",
        task="classification",
    )

    assert result["task"] == "classification"
    assert result["train_rows"] + result["test_rows"] == 60
    assert "logistic_regression" in result["models"]
    assert "random_forest" in result["models"]

    for metrics in result["models"].values():
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_weighted"] <= 1


def test_regression_trains_two_models():
    df = pd.DataFrame({
        "size": list(range(1, 31)),
        "rooms": [2, 3, 4] * 10,
        "price": [100 + size * 20 for size in range(1, 31)],
    })

    result = train_and_compare_models(
        df,
        target_column="price",
        task="regression",
    )

    assert result["task"] == "regression"
    assert "ridge_regression" in result["models"]
    assert "random_forest" in result["models"]

    for metrics in result["models"].values():
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


def test_missing_target_column_raises_error():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "purchased": ["yes", "no", "yes"],
    })

    with pytest.raises(ValueError):
        train_and_compare_models(
            df,
            target_column="unknown",
        )


def test_invalid_task_raises_error():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "purchased": ["yes", "no", "yes"],
    })

    with pytest.raises(ValueError):
        train_and_compare_models(
            df,
            target_column="purchased",
            task="clustering",
        )
