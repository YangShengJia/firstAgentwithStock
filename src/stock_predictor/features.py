from __future__ import annotations

import pandas as pd


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(
    df: pd.DataFrame,
    short_ma_window: int = 5,
    long_ma_window: int = 20,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Create MA5, MA20, RSI, and next-day movement target."""
    featured = df.copy()
    featured["MA5"] = featured["Close"].rolling(window=short_ma_window).mean()
    featured["MA20"] = featured["Close"].rolling(window=long_ma_window).mean()
    featured["RSI"] = calculate_rsi(featured["Close"], window=rsi_window)
    featured["target"] = (featured["Close"].shift(-1) > featured["Close"]).astype(int)

    columns = ["Close", "Volume", "MA5", "MA20", "RSI", "target"]
    return featured[columns].dropna()

