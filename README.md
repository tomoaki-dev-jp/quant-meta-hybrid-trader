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
