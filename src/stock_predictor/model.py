# Train, split, and evaluate the logistic regression stock movement model.
from __future__ import annotations

import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["Close", "Volume", "MA5", "MA20", "RSI"]


def split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split ordered stock data into train and test sets without shuffling."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Not enough rows to create train and test sets.")

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    x_test = test[FEATURE_COLUMNS]
    y_test = test["target"]
    return x_train, x_test, y_train, y_test


def train_model() -> Pipeline:
    """Create a logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate classification metrics."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    if y_test.nunique() < 2:
        roc_auc = math.nan
    else:
        roc_auc = roc_auc_score(y_test, probabilities)

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
    }
