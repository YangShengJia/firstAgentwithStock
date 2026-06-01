from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker!r}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Downloaded data is missing required columns: {missing}")

    df = df.sort_index()
    df.index.name = "Date"
    return df
