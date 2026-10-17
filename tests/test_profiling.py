import pandas as pd
import pytest

from app.tools.profiling import profile_dataset


def test_profile_dataset():
    df = pd.DataFrame({
        "age": [20, 30, None],
        "income": [40000, 60000, 60000],
    })

    result = profile_dataset(df)

    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["columns"][0]["missing_count"] == 1
    assert result["columns"][0]["missing_percentage"] == 33.33
    assert result["duplicate_rows"] == 0


def test_empty_dataset_raises_error():
    with pytest.raises(ValueError):
        profile_dataset(pd.DataFrame())


def test_duplicate_rows_are_detected():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice"],
        "score": [80, 90, 80],
    })

    result = profile_dataset(df)

    assert result["duplicate_rows"] == 1