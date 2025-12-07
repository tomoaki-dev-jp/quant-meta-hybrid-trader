Follow me on [X (Twitter)](https://x.com/x_tomoaki_x)


## yf_ohlcv_history.py とは？

このスクリプトは、Yahoo! Finance から **USD/JPY の5分足ヒストリカルデータを全部ダウンロードして、CSVに保存するツール** だよ。  
通貨ペアや時間足をちょっと書き換えるだけで、他の銘柄にも使い回せるようになってる。

---

## 主な設定パラメータ

- `SYMBOL`  
  - Yahoo! Finance でのティッカー名。  
  - 例:  
    - USD/JPY: `"USDJPY=X"`  
    - EUR/USD: `"EURUSD=X"`  
- `INTERVAL`  
  - ローソクの時間足。`"1m"`, `"5m"`, `"15m"`, `"1h"`, `"1d"` などに変更可能。  
- `PERIOD`  
  - どの期間までさかのぼるか。`"1d"`, `"5d"`, `"1mo"`, `"1y"`, `"max"` など。  
- `OUT_CSV`  
  - 保存されるCSVファイル名。デフォルトだと  
    `yf_USDJPYX_5m_max.csv`  
    みたいな名前になる。

---

## メイン処理の流れ

df = yf.download(
tickers=SYMBOL,
interval=INTERVAL,
period=PERIOD,
auto_adjust=False,
progress=False,
)

1. `yf.download` で指定したティッカー・時間足・期間のOHLCVを取得。  
2. データが空ならメッセージを出して終了。  

---

## カラム名の整形

df = df.rename(
columns={
"Open": "open",
"High": "high",
"Low": "low",
"Close": "close",
"Adj Close": "adj_close",
"Volume": "volume",
}
)


- yfinance が返す列名（`Open`, `High`, ...）を、  
  Python や機械学習コードで扱いやすいように  
  **全部小文字スネークケース** に統一している。  
- `adj_close` が不要ならコメントにある通り削ってもOK。

---

## CSVへの保存

print(df.head())
print(df.tail())
print("rows:", len(df))

df.to_csv(OUT_CSV)
print("saved:", OUT_CSV)


- 先頭・末尾数行と総行数を表示して、ちゃんと取れているか軽くチェック。  
- 最後に `OUT_CSV` で指定したファイル名で **そのままCSV保存**。  
- 生成されたCSVは、さっきの `quant_meta_hybrid_trader.py` の  
  `PAIR_CSV_LIST` にそのまま突っ込んで使える想定。

---

## 使い方まとめ

1. Python環境で `requirements.txt` をインストールしておく。  
2. 必要なら `SYMBOL`, `INTERVAL`, `PERIOD` を好みのものに変更。  
3. スクリプトを実行すると、カレントディレクトリに  
   `yf_〇〇〇_△△_□□.csv` が出力される。  
4. そのCSVを学習・バックテスト用のデータとして使う。

## quant_meta_hybrid_trader.py とは？

このスクリプトは、FXなどの価格CSVを読み込んで  
**「予測モデル（LSTM/Transformer/Regime）＋PPO強化学習エージェント」からなる全部乗せトレード研究フレームワーク** を動かすメインコード。  
完全に「遊び・研究用」で、実運用は非推奨。

---

## ざっくり全体フロー

1. `PAIR_CSV_LIST` に書かれた CSV を読み込む  
2. 価格データから特徴量を作り、  
   - マルチホライズンLSTM  
   - マルチホライズンTransformer  
   - Regime CNN（トレンド/レンジ/高ボラ分類）  
   を学習  
3. それらの出力を使って `HybridEnv`（トレード環境）を構築  
4. PPO でエージェントを学習（簡易メタサーチ付き）  
5. ベスト設定でシミュレーションし、EQカーブを画像として保存  

---

## データ周り：CSV読み込みと特徴量

### 使うCSV

PAIR_CSV_LIST = [
"yf_USDJPYX_5m_max.csv",
# 他ペアを追加可能
]

text

- `yf_ohlcv_history.py` などで作った **OHLCV＋Price列付きCSV** を想定。  
- `load_close_series()` で
  - 変なヘッダ行を除去
  - `Price` → `datetime` にリネームしインデックス化
  - `close` 列をfloat化
  - 必要ならリサンプリング
  を行い、終値の時系列を返す。

### 特徴量生成

returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)

text

- 1ステップリターン
- 12/36本ローリングボラティリティ
- 36本ローリング平均（トレンド）
- RSIを -1〜+1 スケールに正規化
- EMA で平滑化したリターン

これら 6 次元を時系列特徴量として、予測モデルや環境状態に使う。

---

## 予測モデル群

### マルチホライズンLSTM / Transformer

- 入力: 過去 `LSTM_SEQ_LEN` 本分の特徴量系列（6次元）  
- 出力: `FORECAST_HORIZONS = [1,3,6,12,24]` 本先のリターンを同時予測（5次元ベクトル）

class MultiHorizonLSTM(nn.Module): ...
class MultiHorizonTransformer(nn.Module): ...

text

- LSTM は事前に `pretrain_lstm_self_supervised()` で  
  **リターン系列マスク復元タスク** による自己教師ありプリトレを行い、  
  その重みを初期値として使う。

### Regime CNN

class RegimeCNN(nn.Module): ...

text

- 入力: 過去 `REGIME_SEQ_LEN` 本分のリターン窓  
- 出力: 3クラス（Range, Trend, HighVol）の確率  
- ラベルはトレンドとボラをしきい値で判定して自動生成。

### ForecastFusionNet

class ForecastFusionNet(nn.Module):
# LSTM予測 + Transformer予測 + Regime(one-hot) → 統合予測

text

- 入力:  
  - LSTM予測（5次元）  
  - Transformer予測（5次元）  
  - Regime確率（3次元）  
- 出力: 統合された 5 次元リターン予測  
- `HybridEnv` の state 生成時に呼ばれ、  
  「複数モデル＋レジーム情報をまとめた特徴」として使われる。

---

## トレード環境 HybridEnv

class HybridEnv:
# 価格＆特徴量 + 予測フィーダー + OrderBookフック を抱えた環境

text

### 状態ベクトル

`state_dim` は以下を全部つなげた次元数：

- 直近 `STATE_RET_LEN` 本のリターン窓
- 現在の `vol_12`, `trend_36`, `rsi`
- Regime 確率（3次元）
- FusionNet の出力（N_HORIZON）
- 板情報（今はダミーでゼロ3次元）
- 現在ポジション（-1〜1に正規化）

### 行動とポジション

N_ACTIONS = 7 # [-5, -3, -1, 0, 1, 3, 5]

text

- エージェントの行動は「ポジションサイズの選択」。  
- -5〜+5 の7段階（MAX_POSITION=5）で、ロット変更は即時反映。

### 報酬設計

- 生PnL: `position * return`  
- 取引コスト:
  - ボリュームコスト `TRANSACTION_COST * |Δポジション|`
  - 新規ポジション時のみスプレッド＋スリッページを価格換算で控除  
- ネガティブ報酬は `LOSS_FACTOR` 倍で強調  
- `|trend_36|` がしきい値以上なら `TREND_BOOST` をかけてブースト  
- さらに  
  - `lstm_pred[0] * position * LSTM_REWARD_SCALE`  
  - `tf_pred[0]   * position * TF_REWARD_SCALE`  
  のボーナスで「予測方向にポジる」と報酬が増える

---

## PPO エージェント

class ActorCritic(nn.Module): ...

text

- 共有MLP（256→256）から
  - ポリシーヘッド（行動ロジット）
  - バリューヘッド（状態価値）
- `collect_trajectory()` で `HybridEnv` 上の軌跡を生成  
- `compute_gae()` で GAE による advantage/return を計算  
- クリッピング付きPPO損失＋価値損失＋エントロピー正則化で学習。

---

## メタサーチ（ハイパラ探索）

def meta_search(envs: List[HybridEnv], trials: int = 5):

text

- ランダムサーチするパラメータ：
  - 学習率 `lr`
  - 損失強調 `loss_factor`
  - トレンドブースト `trend_boost`
- 各設定で `train_with_config()` を回し、  
  最後の数エピソードの平均リターンが最も高いモデルと設定を採用。

---

## シミュレーションとEQカーブ出力

def run_simulation(env: HybridEnv, model: ActorCritic,
steps: int = 500, log_interval: int = 50):

text

- 学習済みモデルで greedy 行動（argmax）をとり続け、  
  `equity *= (1 + reward)` で資産曲線を更新。  
- `simulation_equity_curve.png` に EQカーブを保存し、  
  ログに t, action, position, reward, equity を出力。

---

## main() の実行手順

if name == "main":
main()

text

1. `PAIR_CSV_LIST` の各CSVを読み込み  
2. 各ペアごとに forecaster（LSTM/TF）＋ RegimeCNN を学習  
3. それらを使って `HybridEnv` を構築  
4. 全envを使ってメタサーチ付きPPO学習  
5. ベストモデルで代表ペアのシミュレーションを走らせ、EQカーブを保存  

---

## 注意事項

- このフレームワークは **完全に研究・遊び専用**。  
  本物の資金での運用は禁止。  
- マーケットインパクトや約定リスクなどはかなり単純化しているので、  
  得られた戦略をそのまま実戦投入すると死ぬ可能性が高い。  
- 学習期間・テスト期間を明確に分けて、  
  過学習のチェックを行うことを強く推奨。

## What is `yf_ohlcv_history.py`?

This script is a small tool that **downloads all historical 5‑minute OHLCV data for USD/JPY from Yahoo! Finance and saves it as a CSV file**.  
By tweaking the symbol and interval a bit, you can easily reuse it for other instruments.

---

## Main configuration parameters

- `SYMBOL`  
  - Ticker name on Yahoo! Finance.  
  - Examples:  
    - USD/JPY: `"USDJPY=X"`  
    - EUR/USD: `"EURUSD=X"`  
- `INTERVAL`  
  - Timeframe of the candlestick bars. Can be changed to `"1m"`, `"5m"`, `"15m"`, `"1h"`, `"1d"`, etc.  
- `PERIOD`  
  - How far back to download. Can be `"1d"`, `"5d"`, `"1mo"`, `"1y"`, `"max"`, etc.  
- `OUT_CSV`  
  - Output CSV filename. By default it becomes something like  
    `yf_USDJPYX_5m_max.csv`.

---

## Main processing flow

df = yf.download(
tickers=SYMBOL,
interval=INTERVAL,
period=PERIOD,
auto_adjust=False,
progress=False,
)

text

1. Use `yf.download` to fetch OHLCV for the specified ticker, interval, and period.  
2. If the returned data is empty, print a message and exit.  

---

## Normalizing column names

df = df.rename(
columns={
"Open": "open",
"High": "high",
"Low": "low",
"Close": "close",
"Adj Close": "adj_close",
"Volume": "volume",
}
)

text

- Column names returned by yfinance (`Open`, `High`, …) are converted to  
  **lower‑case snake_case** so they are easier to work with in Python and ML code.  
- If you do not need `adj_close`, you can simply drop it as the comment suggests.

---

## Saving to CSV

print(df.head())
print(df.tail())
print("rows:", len(df))

df.to_csv(OUT_CSV)
print("saved:", OUT_CSV)

text

- Prints the first and last few rows plus the total row count for a quick sanity check.  
- Then saves the data directly to the file specified by `OUT_CSV`.  
- This CSV is designed to be used as an input file in `quant_meta_hybrid_trader.py` via `PAIR_CSV_LIST`.

---

## Usage summary

1. Install the Python dependencies from `requirements.txt`.  
2. Change `SYMBOL`, `INTERVAL`, and `PERIOD` as you like.  
3. Run the script; a file named like `yf_<symbol>_<interval>_<period>.csv` will be created in the current directory.  
4. Use that CSV as training / backtest data.

---

## What is `quant_meta_hybrid_trader.py`?

This script reads FX (or other) price CSVs and runs a  
**“kitchen‑sink” trading research framework combining prediction models (LSTM / Transformer / Regime) and a PPO reinforcement‑learning agent**.  
It is purely for experimentation and research; it is **not** intended for live trading.

---

## High‑level workflow

1. Load the CSV files listed in `PAIR_CSV_LIST`.  
2. Build features from price data and train:  
   - Multi‑horizon LSTM  
   - Multi‑horizon Transformer  
   - Regime CNN (classifying Trend / Range / HighVol)  
3. Construct a `HybridEnv` trading environment that uses the models’ outputs.  
4. Train an agent with PPO, with a simple random‑search style meta‑tuning of some hyperparameters.  
5. Run a simulation with the best configuration and save the resulting equity curve as an image.

---

## Data handling: CSV loading and features

### Input CSV

PAIR_CSV_LIST = [
"yf_USDJPYX_5m_max.csv",
# add more pairs here if you like
]

text

- Assumes CSVs generated by something like `yf_ohlcv_history.py`,  
  containing OHLCV plus a `Price` column.  
- `load_close_series()` will:
  - remove weird header rows  
  - rename `Price` → `datetime` and set it as the index  
  - convert the `close` column to float  
  - optionally resample if configured  
  and finally returns a clean close‑price time series.

### Feature generation

returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)

text

- 1‑step returns  
- 12 / 36 bar rolling volatility  
- 36 bar rolling mean (trend)  
- RSI normalized to the range approximately \[-1, 1\]  
- EMA‑smoothed returns  

These six dimensions form the time‑series features used by the prediction models and by the environment.

---

## Prediction models

### Multi‑horizon LSTM / Transformer

- Input: past `LSTM_SEQ_LEN` bars of 6‑dimensional features.  
- Output: a 5‑dimensional vector predicting returns for horizons  
  `FORECAST_HORIZONS = [1, 3, 6, 12, 24]`.

class MultiHorizonLSTM(nn.Module): ...
class MultiHorizonTransformer(nn.Module): ...

text

- The LSTM is first **self‑supervised pre‑trained** via `pretrain_lstm_self_supervised()`  
  on a masked‑value reconstruction task on the returns series,  
  and these weights are then used as initialization for the forecaster LSTM.

### Regime CNN

class RegimeCNN(nn.Module): ...

text

- Input: a window of length `REGIME_SEQ_LEN` of returns.  
- Output: probabilities over three regimes (Range, Trend, HighVol).  
- Labels are generated automatically using simple thresholds on trend and volatility.

### ForecastFusionNet

class ForecastFusionNet(nn.Module):
# LSTM forecast + Transformer forecast + Regime(one-hot) → fused forecast

text

- Inputs:  
  - LSTM forecast (5D)  
  - Transformer forecast (5D)  
  - Regime probabilities (3D)  
- Output: fused 5D return forecast.  
- Called when building the state in `HybridEnv`, so the agent sees a combined representation of all models and the regime.

---

## Trading environment: `HybridEnv`

class HybridEnv:
# Holds prices, features, forecast feeders and an orderbook hook

text

### State vector

`state_dim` is the concatenation of:

- Recent returns window of length `STATE_RET_LEN`  
- Current `vol_12`, `trend_36`, `rsi`  
- Regime probabilities (3D)  
- Fused forecast from `ForecastFusionNet` (N_HORIZON)  
- Orderbook features (currently a 3D zero vector)  
- Current position, normalized to \[-1, 1\]

### Actions and position

N_ACTIONS = 7 # [-5, -3, -1, 0, 1, 3, 5]

text

- The agent’s action is the **position size**.  
- One of seven discrete values from -5 to +5 (`MAX_POSITION=5`),  
  and changes are applied immediately.

### Reward design

- Raw PnL: `position * return`  
- Trading costs:
  - Volume cost: `TRANSACTION_COST * |Δposition|`  
  - Spread + slippage, charged only when a new position is opened, converted into return units  
- Negative rewards are amplified by `LOSS_FACTOR`.  
- If `|trend_36|` exceeds a threshold, the reward is multiplied by `TREND_BOOST`.  
- Additionally:
  - `lstm_pred[0] * position * LSTM_REWARD_SCALE`  
  - `tf_pred[0]   * position * TF_REWARD_SCALE`  
  provide extra reward when the position is aligned with the direction of the forecasts.

---

## PPO agent

class ActorCritic(nn.Module): ...

text

- Shared MLP (256 → 256) feeding into:
  - Policy head (action logits)  
  - Value head (state value)  
- `collect_trajectory()` generates rollouts on `HybridEnv`.  
- `compute_gae()` computes advantages and returns using GAE.  
- Training uses clipped PPO loss + value loss + entropy regularization.

---

## Meta‑search (hyperparameter tuning)

def meta_search(envs: List[HybridEnv], trials: int = 5):

text

- Randomly samples:
  - Learning rate `lr`  
  - Loss amplification `loss_factor`  
  - Trend boost factor `trend_boost`  
- For each config, runs `train_with_config()` and  
  keeps the model whose last few episode rewards have the highest mean.

---

## Simulation and equity‑curve output

def run_simulation(env: HybridEnv, model: ActorCritic,
steps: int = 500, log_interval: int = 50):

text

- Runs the trained model in greedy mode (argmax over logits).  
- Updates the equity with `equity *= (1 + reward)` each step.  
- Saves the equity curve as `simulation_equity_curve.png` and  
  logs `t, action, position, reward, equity`.

---

## Running `main()`

if name == "main":
main()

text

1. Load each CSV in `PAIR_CSV_LIST`.  
2. Train forecasters (LSTM / Transformer) and `RegimeCNN` per pair.  
3. Build a `HybridEnv` for each instrument.  
4. Run PPO training with meta‑search across environments.  
5. Run a simulation on a representative pair and save the equity curve.

---

## Important notes

- This framework is **strictly for research and experimentation only**.  
  It must **not** be used with real money.  
- Market impact, execution risk, and many other real‑world effects are heavily simplified,  
  so applying any discovered strategy directly to live trading is very dangerous.  
- Always split data into clear training and testing periods  
  and check for overfitting before drawing any conclusions.