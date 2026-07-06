# Command-line entry point that downloads data, trains the model, and saves metrics.
from __future__ import annotations

import argparse
import json
import matplotlib.pyplot as plt

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
    parser.add_argument("--debug", action="store_true", help="Print debug information.")

    parser.add_argument("--debug-table", 
                        choices=["features"], 
                        default=None, 
                        help="Choose which debug table to print.", )
    parser.add_argument("--ma5", action="store_true", help="Show MA5 column.")
    parser.add_argument("--ma20", action="store_true", help="Show MA20 column.")
    parser.add_argument("--rsi", action="store_true", help="Show RSI column.")
    parser.add_argument("--debug-rows", 
                        type=int, 
                        default=100, 
                        help="Number of rows to show in debug plot.", )

    return parser.parse_args()

def get_debug_feature_columns(args) -> list[str]:
    columns = ["Close"]

    if args.ma5:
        columns.append("MA5")

    if args.ma20:
        columns.append("MA20")

    if args.rsi:
        columns.append("RSI")

    # 如果都沒指定，就預設全顯示
    if columns == ["Close"]:
        columns = ["Close", "MA5", "MA20", "RSI"]

    return columns

def plot_feature_debug(dataset, columns: list[str], rows: int = 100) -> None:
    plot_data = dataset[columns].head(rows)

    plt.figure(figsize=(12, 6))

    for column in columns:
        plt.plot(plot_data.index, plot_data[column], label=column)

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Feature Debug Plot")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        test_size=args.test_size,
    )

    raw_data = download_stock_data(config.ticker, config.start_date, config.end_date)
    if args.debug:
        print("CCC raw_data.head():")
        print(raw_data.head().to_string())
    dataset = build_features(
        raw_data,
        short_ma_window=config.short_ma_window,
        long_ma_window=config.long_ma_window,
        rsi_window=config.rsi_window,
        debug=args.debug,
    )

    if args.debug and args.debug_table == "features":
        selected_columns = get_debug_feature_columns(args)
        plot_feature_debug(dataset, selected_columns, rows=args.debug_rows)

    x_train, x_test, y_train, y_test = split_time_series(dataset, test_size=config.test_size, debug=args.debug,)
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
