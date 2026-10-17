
import pandas as pd
import pytest

from app.tools.eda import perform_eda


def test_eda_generates_statistics_and_plots(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "salary": [30000, 40000, 50000, 60000],
        "city": ["Pune", "Mumbai", "Pune", "Delhi"],
    })

    result = perform_eda(df, output_dir=str(tmp_path))

    assert result["row_count"] == 4
    assert result["column_count"] == 3
    assert "age" in result["numeric_summary"]
    assert "city" in result["categorical_summary"]
    assert "age" in result["correlations"]
    assert len(result["plot_files"]) == 2

    for plot_file in result["plot_files"]:
        assert (tmp_path / plot_file.split("/")[-1]).exists()


def test_eda_rejects_empty_dataset():
    with pytest.raises(ValueError):
        perform_eda(pd.DataFrame())
