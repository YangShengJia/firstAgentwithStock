# Build moving average, RSI, and next-day movement target features.
from __future__ import annotations

import pandas as pd


def calculate_rsi(close: pd.Series, window: int = 14, debug: bool = False,) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    if debug:
        print("DDD delta.head():")
        print(delta.head().to_string())
        print("EEE gain.head():")
        print(gain.head().to_string())
        print("FFF loss.head():")
        print(loss.head().to_string())
        print("GGG len(avg_gain):")
        print(len(avg_gain))
        print("HHH avg_loss.head():")
        print(avg_loss.head().to_string())
    return rsi.fillna(50)


def build_features(
    df: pd.DataFrame,
    short_ma_window: int = 5,
    long_ma_window: int = 20,
    rsi_window: int = 14,
    debug: bool = False,
) -> pd.DataFrame:
    """Create MA5, MA20, RSI, and next-day movement target."""
    featured = df.copy()
    featured["MA5"] = featured["Close"].rolling(window=short_ma_window).mean()
    featured["MA20"] = featured["Close"].rolling(window=long_ma_window).mean()
    featured["RSI"] = calculate_rsi(featured["Close"], window=rsi_window, debug=debug)
    # Build next-day close first so rows without future prices can be excluded.
    featured["next_close"] = featured["Close"].shift(-1)

    # Drop rows that cannot be labeled, usually the final row.
    featured = featured.dropna(subset=["next_close"])

    # Target: 1 if the next close is higher than today's close, otherwise 0.
    featured["target"] = (featured["next_close"] > featured["Close"]).astype(int)


    columns = ["Close", "Volume", "MA5", "MA20", "RSI", "target"]
    return featured[columns].dropna()
