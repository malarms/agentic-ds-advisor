
import pandas as pd


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Identify common data-quality issues in a DataFrame."""

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    missing_values = {}

    for column in df.columns:
        count = int(df[column].isna().sum())

        if count > 0:
            missing_values[str(column)] = {
                "count": count,
                "percentage": round(
                    float(df[column].isna().mean() * 100), 2
                ),
            }

    constant_columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    duplicate_rows = int(df.duplicated().sum())

    issues = []

    if missing_values:
        issues.append("Missing values detected.")

    if duplicate_rows:
        issues.append("Duplicate rows detected.")

    if constant_columns:
        issues.append("Constant columns detected.")

    if not issues:
        issues.append("No basic data-quality issues detected.")

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "issues": issues,
    }