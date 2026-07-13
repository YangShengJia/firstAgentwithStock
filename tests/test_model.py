import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from stock_predictor.model import train_model


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
