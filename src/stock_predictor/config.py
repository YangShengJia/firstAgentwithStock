from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    ticker: str = "2330.TW"
    start_date: str = "2015-01-01"
    end_date: str | None = None
    test_size: float = 0.2
    rsi_window: int = 14
    short_ma_window: int = 5
    long_ma_window: int = 20

