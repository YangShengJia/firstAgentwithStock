# Train, split, and evaluate the logistic regression stock movement model.
from __future__ import annotations

import math

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["Close", "Volume", "MA5", "MA20", "RSI"]
MODEL_CHOICES = ("logistic", "random-forest", "gradient-boosting")


def split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.2,
    debug: bool = False,
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
    if debug:
        print(f"pd.DataFrame Total rows: {len(df)}")
        print(f"Split index: {split_index}")
        print(f"Train rows: {len(train)}")
        print(f"Test rows: {len(test)}")
        print(f"First train row:\n{train.iloc[[0]]}")
        print(f"First test row:\n{test.iloc[[0]]}")
    return x_train, x_test, y_train, y_test


def train_model(model_name: str = "logistic"):
    """Create a classifier by model name."""
    if model_name == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
    if model_name == "random-forest":
        return RandomForestClassifier(random_state=42)
    if model_name == "gradient-boosting":
        return GradientBoostingClassifier(random_state=42)

    supported_models = ", ".join(MODEL_CHOICES)
    raise ValueError(f"Unsupported model_name: {model_name}. Choose one of: {supported_models}.")


def evaluate_model(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate classification metrics."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    positive_rate = float(y_test.mean())
    negative_rate = float(1 - positive_rate)

    accuracy = accuracy_score(y_test, predictions)
    always_up_accuracy = y_test.mean()
    majority_baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    if y_test.nunique() < 2:
        roc_auc = math.nan
    else:
        roc_auc = roc_auc_score(y_test, probabilities)

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "always_up_accuracy": float(always_up_accuracy),
        "majority_baseline_accuracy": float(majority_baseline_accuracy),
    }
