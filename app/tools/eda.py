
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def perform_eda(df: pd.DataFrame, output_dir: str = "reports") -> dict:
    """Generate descriptive statistics, correlations and plots."""

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    import os

    os.makedirs(output_dir, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    numeric_summary = {}

    if not numeric_df.empty:
        numeric_summary = (
            numeric_df.describe()
            .round(3)
            .to_dict()
        )

    categorical_summary = {}

    for column in categorical_df.columns:
        categorical_summary[str(column)] = {
            str(key): int(value)
            for key, value in (
                categorical_df[column]
                .value_counts(dropna=False)
                .head(10)
                .items()
            )
        }

    correlations = {}

    if numeric_df.shape[1] >= 2:
        correlations = (
            numeric_df.corr()
            .round(3)
            .to_dict()
        )

    plot_files = []

    for column in numeric_df.columns[:5]:
        filename = f"{output_dir}/{column}_histogram.png"

        plt.figure(figsize=(8, 5))
        numeric_df[column].dropna().hist(bins=20)
        plt.title(f"Distribution of {column}")
        plt.xlabel(str(column))
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        plot_files.append(filename)

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "plot_files": plot_files,
    }