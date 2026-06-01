from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_predictor.config import ProjectConfig
from stock_predictor.data import download_stock_data
from stock_predictor.features import build_features
from stock_predictor.model import evaluate_model, split_time_series, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict TSMC next-day stock movement.")
    parser.add_argument("--ticker", default=ProjectConfig.ticker, help="Yahoo Finance ticker.")
    parser.add_argument("--start", default=ProjectConfig.start_date, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=ProjectConfig.end_date, help="End date, YYYY-MM-DD.")
    parser.add_argument("--test-size", type=float, default=ProjectConfig.test_size)
    parser.add_argument("--metrics-path", default="reports/metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        test_size=args.test_size,
    )

    raw_data = download_stock_data(config.ticker, config.start_date, config.end_date)
    dataset = build_features(
        raw_data,
        short_ma_window=config.short_ma_window,
        long_ma_window=config.long_ma_window,
        rsi_window=config.rsi_window,
    )

    x_train, x_test, y_train, y_test = split_time_series(dataset, test_size=config.test_size)
    model = train_model()
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    metrics.update(
        {
            "ticker": config.ticker,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        }
    )

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Ticker: {metrics['ticker']}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

