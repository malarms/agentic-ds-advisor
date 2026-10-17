
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


def train_and_compare_models(
    df: pd.DataFrame,
    target_column: str,
    task: str = "classification",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train baseline models and compare held-out test performance."""

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    if task not in {"classification", "regression"}:
        raise ValueError(
            "task must be 'classification' or 'regression'."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = df.dropna(subset=[target_column]).copy()

    if data.empty:
        raise ValueError("The target column has no usable values.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    if task == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least 2 classes.")

        if y.value_counts().min() < 2:
            raise ValueError(
                "Each class needs at least 2 rows for a stratified split."
            )

        stratify = y
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
            ),
        }
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target.")

        if len(data) < 3:
            raise ValueError("Regression requires at least 3 usable rows.")

        stratify = None
        models = {
            "ridge_regression": Ridge(),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
            ),
        }

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append((
            "numeric", numeric_pipeline, numeric_columns
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append((
            "categorical", categorical_pipeline, categorical_columns
        ))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    results = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        if task == "classification":
            metrics = {
                "accuracy": round(
                    float(accuracy_score(y_test, predictions)), 4
                ),
                "f1_weighted": round(
                    float(f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )),
                    4,
                ),
            }
        else:
            metrics = {
                "mae": round(
                    float(mean_absolute_error(y_test, predictions)), 4
                ),
                "rmse": round(
                    float(mean_squared_error(
                        y_test, predictions
                    ) ** 0.5),
                    4,
                ),
                "r2": round(
                    float(r2_score(y_test, predictions)), 4
                ),
            }

        results[model_name] = metrics

    return {
        "task": task,
        "target_column": target_column,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": results,
        "note": (
            "Metrics are from a single held-out test split. "
            "They are preliminary estimates, not a guarantee "
            "of future performance."
        ),
    }