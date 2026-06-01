import pandas as pd

from stock_predictor.features import build_features, calculate_rsi


def test_calculate_rsi_returns_expected_range():
    close = pd.Series([100, 101, 102, 101, 103, 104, 102, 105, 106, 107, 108, 106, 109, 110, 111])

    rsi = calculate_rsi(close, window=5)

    assert rsi.between(0, 100).all()


def test_build_features_creates_expected_columns_and_target():
    df = pd.DataFrame(
        {
            "Open": range(1, 31),
            "High": range(2, 32),
            "Low": range(0, 30),
            "Close": range(1, 31),
            "Volume": [1000] * 30,
        }
    )

    result = build_features(df)

    assert list(result.columns) == ["Close", "Volume", "MA5", "MA20", "RSI", "target"]
    assert result["target"].isin([0, 1]).all()
    assert len(result) > 0

