<!-- Project overview and usage guide for the TSMC stock movement predictor. -->

# firstAgentwithStock

使用 Python、yfinance 與 Logistic Regression 建立台積電隔日漲跌預測專案。

## 功能

- 使用 `yfinance` 下載台積電股價資料，預設 ticker 為 `2330.TW`
- 建立技術指標特徵：`MA5`、`MA20`、`RSI`
- 預測隔日收盤價相較今日是上漲或下跌
- 使用 `LogisticRegression` 訓練分類模型
- 輸出 `Accuracy` 與 `ROC-AUC`
- 支援 `--debug` 檢視資料流程，並可用 `matplotlib` 顯示 `Close`、`MA5`、`MA20`、`RSI` 圖形
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

建議先建立並啟動虛擬環境，再安裝本地 package。

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 執行

```bash
python -m stock_predictor.main
```

預設會使用：

```text
ticker: 2330.TW
start date: 2020-01-01
test size: 0.2
metrics path: reports/metrics.json
```

執行後會在終端機輸出：

- Accuracy
- ROC-AUC
- 訓練資料筆數
- 測試資料筆數

同時會將結果寫入 `reports/metrics.json`。

## 執行範例結果

```bash
python -m stock_predictor.main --ticker 2434.TW --start 2015-01-01
```

範例輸出：

```text
Ticker: 2434.TW
Train rows: 2221
Test rows: 556
Accuracy: 0.5144
ROC-AUC: 0.5149
```

輸出解釋：

- `Ticker: 2434.TW`：這次執行使用的 Yahoo Finance 股票代號。
- `Train rows: 2221`：用來訓練模型的資料筆數。時間序列切分會使用前段資料作為訓練集。
- `Test rows: 556`：用來評估模型的資料筆數。時間序列切分會使用後段資料作為測試集。
- `Accuracy: 0.5144`：模型預測隔日漲跌方向的正確率，約等於 51.44%。
- `ROC-AUC: 0.5149`：衡量模型區分上漲與未上漲樣本的能力，越接近 1 越好，`0.5` 附近代表接近隨機猜測。

這組結果只代表此 ticker、日期範圍、特徵與模型設定下的回測結果，不代表未來投資績效。

## CLI 參數

| 參數 | 型別 / 用法 | 預設值 | 用途 |
| --- | --- | --- | --- |
| `--ticker` | 文字，例如 `2330.TW` | `2330.TW` | 指定 Yahoo Finance ticker |
| `--start` | 日期，例如 `2020-01-01` | `2020-01-01` | 指定下載資料開始日期 |
| `--end` | 日期，例如 `2026-06-02` | `None` | 指定下載資料結束日期；不填則抓到可取得的最新資料 |
| `--test-size` | 小數，例如 `0.2` | `0.2` | 指定測試集比例；`0.2` 代表後 20% 當 test |
| `--metrics-path` | 路徑，例如 `reports/metrics.json` | `reports/metrics.json` | 指定評估結果輸出位置 |
| `--debug` | 開關，不需要給值 | `False` | 開啟 debug 模式 |
| `--debug-table` | 目前支援 `features` | `None` | 指定是否顯示 features debug 圖 |
| `--ma5` | 開關，不需要給值 | `False` | 在 features debug 圖中顯示 `MA5` |
| `--ma20` | 開關，不需要給值 | `False` | 在 features debug 圖中顯示 `MA20` |
| `--rsi` | 開關，不需要給值 | `False` | 在 features debug 圖中顯示 `RSI` |
| `--debug-rows` | 整數，例如 `100` | `100` | 指定 debug 圖要顯示前幾筆資料 |

## 常用指令

指定股票與日期：

```bash
python -m stock_predictor.main --ticker 2330.TW --start 2020-01-01 --end 2026-06-02
```

指定 train/test 切分比例：

```bash
python -m stock_predictor.main --test-size 0.2
```

`test-size = 0.2` 代表前 80% 當訓練集，後 20% 當測試集。因為是時間序列資料，所以不進行 random shuffle。

## Debug 與資料視覺化

開啟 debug 模式：

```bash
python -m stock_predictor.main --debug
```

`--debug` 會在終端機輸出 raw data、RSI 中間值、train/test split 等資訊，方便理解資料流程。

顯示 features 圖形：

```bash
python -m stock_predictor.main --debug --debug-table features
```

如果沒有指定 `--ma5`、`--ma20`、`--rsi`，圖形會預設顯示：

```text
Close, MA5, MA20, RSI
```

只顯示 `MA5`：

```bash
python -m stock_predictor.main --debug --debug-table features --ma5
```

只顯示 `MA20`：

```bash
python -m stock_predictor.main --debug --debug-table features --ma20
```

只顯示 `RSI`：

```bash
python -m stock_predictor.main --debug --debug-table features --rsi
```

同時顯示多個指標：

```bash
python -m stock_predictor.main --debug --debug-table features --ma5 --ma20 --rsi
```

指定圖形顯示筆數：

```bash
python -m stock_predictor.main --debug --debug-table features --ma5 --debug-rows 200
```

`--debug-rows 200` 代表只取前 200 筆資料來畫圖。`--debug-rows` 是數值參數，不是開關。

注意：`RSI` 的尺度是 0 到 100；`Close`、`MA5`、`MA20` 是股價尺度。放在同一張圖時，`RSI` 可能看起來貼近底部。

## 模型說明

目標欄位 `target` 定義如下：

```text
隔日收盤價 > 今日收盤價 => 1，上漲
隔日收盤價 <= 今日收盤價 => 0，下跌或持平
```

沒有隔日收盤價的資料列會先被排除，避免最後一筆資料因為 `NaN` 比較而被錯誤標成 `0`。

資料切分採用時間序列常見作法，依日期排序後以前 80% 作為訓練集，後 20% 作為測試集，不進行隨機打散。

## 常見問題

### `unrecognized arguments: --ma5`

請確認指令使用的是長參數：

```bash
--ma5
```

不是：

```bash
-ma5
```

### `matplotlib` 找不到

請確認已安裝專案依賴：

```bash
python -m pip install -e .
```

或直接安裝：

```bash
python -m pip install matplotlib
```

如果 VS Code 顯示無法解析 `matplotlib.pyplot`，也要確認 VS Code 選到正確的 Python interpreter。

## 測試

```bash
pytest
```
