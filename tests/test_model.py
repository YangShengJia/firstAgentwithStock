import pytest
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from stock_predictor.model import evaluate_model, split_time_series, train_model


def test_train_model_defaults_to_logistic_pipeline():
    model = train_model()

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)


def test_train_model_supports_random_forest():
    model = train_model("random-forest")

    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 42


def test_train_model_supports_gradient_boosting():
    model = train_model("gradient-boosting")

    assert isinstance(model, GradientBoostingClassifier)
    assert model.random_state == 42


def test_train_model_rejects_unknown_model():
    with pytest.raises(ValueError):
        train_model("unsupported-model")


def test_split_time_series_preserves_time_order():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Close": range(10),
            "Volume": [1000] * 10,
            "MA5": range(10),
            "MA20": range(10),
            "RSI": [50] * 10,
            "target": [0, 1] * 5,
        },
        index=index,
    )

    x_train, x_test, y_train, y_test = split_time_series(df, test_size=0.2)

    assert list(x_train.index) == list(index[:8])
    assert list(x_test.index) == list(index[8:])
    assert list(y_train.index) == list(index[:8])
    assert list(y_test.index) == list(index[8:])


def test_evaluate_model_returns_baseline_metrics():
    class FakeModel:
        def predict(self, x_test):
            return [1, 0, 1, 0]

        def predict_proba(self, x_test):
            return pd.DataFrame({0: [0.2, 0.8, 0.3, 0.7], 1: [0.8, 0.2, 0.7, 0.3]}).to_numpy()

    x_test = pd.DataFrame(
        {
            "Close": [10, 11, 12, 13],
            "Volume": [1000, 1000, 1000, 1000],
            "MA5": [10, 10, 11, 12],
            "MA20": [10, 10, 11, 12],
            "RSI": [50, 55, 60, 45],
        }
    )
    y_test = pd.Series([1, 0, 0, 0])

    metrics = evaluate_model(FakeModel(), x_test, y_test)

    assert {
        "accuracy",
        "roc_auc",
        "positive_rate",
        "negative_rate",
        "always_up_accuracy",
        "majority_baseline_accuracy",
    }.issubset(metrics)
