# firstAgentwithStock

使用 Python、yfinance 與 Logistic Regression 建立台積電隔日漲跌預測專案。

## 功能

- 使用 `yfinance` 下載台積電股價資料，預設 ticker 為 `2330.TW`
- 建立技術指標特徵：`MA5`、`MA20`、`RSI`
- 預測隔日收盤價相較今日是上漲或下跌
- 使用 `LogisticRegression` 訓練分類模型
- 輸出 `Accuracy` 與 `ROC-AUC`
- 專案結構清楚，方便後續擴充

## 專案結構

```text
firstAgentwithStock/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── stock_predictor/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── main.py
│       └── model.py
└── tests/
    └── test_features.py
```

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 執行

```bash
python -m stock_predictor.main
```

也可以指定下載期間：

```bash
python -m stock_predictor.main --start 2015-01-01 --end 2026-06-02
```

執行後會在終端機輸出：

- Accuracy
- ROC-AUC
- 訓練資料筆數
- 測試資料筆數

同時會將結果寫入 `reports/metrics.json`。

## 模型說明

目標欄位 `target` 定義如下：

```text
隔日收盤價 > 今日收盤價 => 1，上漲
隔日收盤價 <= 今日收盤價 => 0，下跌或持平
```

資料切分採用時間序列常見作法，依日期排序後以前 80% 作為訓練集，後 20% 作為測試集，不進行隨機打散。

## 測試

```bash
pytest
```

