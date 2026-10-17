
import pandas as pd
import pytest

from app.tools.data_quality import analyze_data_quality


def test_detects_missing_values():
    df = pd.DataFrame({
        "age": [20, None, 30],
        "score": [80, 90, 100],
    })

    result = analyze_data_quality(df)

    assert result["missing_values"]["age"]["count"] == 1
    assert result["missing_values"]["age"]["percentage"] == 33.33


def test_detects_duplicate_rows():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice"],
        "score": [80, 90, 80],
    })

    result = analyze_data_quality(df)

    assert result["duplicate_rows"] == 1


def test_detects_constant_columns():
    df = pd.DataFrame({
        "city": ["Pune", "Pune", "Pune"],
        "age": [20, 30, 40],
    })

    result = analyze_data_quality(df)

    assert "city" in result["constant_columns"]


def test_empty_dataset_raises_error():
    with pytest.raises(ValueError):
        analyze_data_quality(pd.DataFrame())
