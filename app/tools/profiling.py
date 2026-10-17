
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a structured profile of a pandas DataFrame."""

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    columns = []

    for column in df.columns:
        series = df[column]

        columns.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100), 2
            ),
            "unique_count": int(series.nunique()),
        })

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": columns,
        "duplicate_rows": int(df.duplicated().sum()),
    }
