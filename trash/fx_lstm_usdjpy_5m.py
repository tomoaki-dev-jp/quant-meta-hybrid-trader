# fx_lstm_usdjpy_1m_best.py
#
# yfinance の yf_USDJPYX_1m_max.csv を元に
# 多特徴量 LSTM で USD/JPY 1分足の未来1ステップを予測する最強構成。

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============ 設定 ============

CSV_FILE     = "fx/yf_USDJPYX_5m_max.csv"  # yfinanceで落とした1分足CSV
USE_RESAMPLE = False                    # True にすると 5分足などに変換
RESAMPLE_RULE = "5min"                  # "5min", "15min", "1H" など

SEQ_LEN      = 72      # 過去何本を見るか（1mなら180=過去3時間）
HORIZON      = 3        # 何ステップ先を予測するか（1mなら1分先）
TRAIN_RATIO  = 0.8      # 学習データの割合
EPOCHS       = 300      # ガチ学習用（GPU推奨）
BATCH_SIZE   = 128
LR           = 1e-4
MAX_POINTS   = 20000    # データが多すぎると重いので上限

# ============ モデル構成（LSTMの太さ） ============
HIDDEN_SIZE  = 128
NUM_LAYERS   = 3

# ============ デバイス判定 ============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)

# ============ CSVローダ ============

def load_yf_1m_csv(csv_file: str) -> pd.DataFrame:
    """
    yfinance の 1m CSV フォーマットを読み込み、
    datetime index + columns: open, high, low, close, volume のDataFrameを返す。
    形式例:
    Price,adj_close,close,high,low,open,volume
    Ticker,USDJPY=X,...
    Datetime,,,,,,
    2025-11-24 00:00:00+00:00,156.66,...
    """
    df = pd.read_csv(csv_file)

    if "Price" not in df.columns:
        raise ValueError("CSVに 'Price' 列がありません。フォーマットを確認してください。")

    # Ticker/Datetime行を除外（Price列に "Ticker" や "Datetime" を含む行）
    mask_bad = df["Price"].astype(str).str.contains("Ticker|Datetime", na=False)
    df = df[~mask_bad].copy()

    # Price列をdatetimeにする
    df = df.rename(columns={"Price": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # 必須列確認
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSVに '{col}' 列がありません。")

    # 数値化
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    print("Loaded raw OHLCV rows:", len(df))

    # 任意でリサンプリング（5分足など）
    if USE_RESAMPLE:
        print(f"Resampling to: {RESAMPLE_RULE}")
        df = df[["open", "high", "low", "close", "volume"]].resample(
            RESAMPLE_RULE
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        print("Resampled rows:", len(df))

    # 古い分を切り捨てて上限を守る
    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]
        print(f"Truncated to last {MAX_POINTS} rows.")

    return df


# ============ 特徴量生成 ============

def build_features(df: pd.DataFrame):
    """
    df: index=datetime, columns: open, high, low, close, volume
    戻り値:
      feat_norm: 正規化済み特徴量 (N, F)
      target_norm: 正規化済みターゲット (N,)
      target_mean, target_std: 逆正規化用パラメータ
    """
    feat_df = pd.DataFrame(index=df.index)

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low  = df["low"]
    vol  = df["volume"]

    # 1. close 自体
    feat_df["close"] = close

    # 2. 1本差分リターン（パーセンテージ）
    feat_df["ret1"] = close.pct_change().fillna(0.0)

    # 3. 高値-安値のスプレッド
    feat_df["hl_spread"] = (high - low) / (close + 1e-9)

    # 4. 始値→終値の変化
    feat_df["oc_change"] = (close - open_) / (open_ + 1e-9)

    # 5. 生の volume
    feat_df["volume"] = vol

    # 6. ローリングボラティリティ（短期）
    vol_window = 30  # 30本 = 30分
    feat_df["vol_roll"] = close.pct_change().rolling(vol_window).std().fillna(0.0)

    # 欠損/無限大を除去
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

    # ターゲット（close）を用意
    target = feat_df["close"].values.astype(np.float32)
    target_mean = target.mean()
    target_std  = target.std() if target.std() > 0 else 1.0
    target_norm = (target - target_mean) / target_std

    # 各特徴量を列ごとに標準化
    feat_values = feat_df.values.astype(np.float32)  # (N, F)
    feat_mean = feat_values.mean(axis=0, keepdims=True)
    feat_std  = feat_values.std(axis=0, keepdims=True)
    feat_std[feat_std == 0] = 1.0
    feat_norm = (feat_values - feat_mean) / feat_std

    return feat_df.index, feat_norm, target_norm, target_mean, target_std


# ============ シーケンス作成 ============

def create_sequences_multi(features: np.ndarray, target: np.ndarray,
                           seq_len: int, horizon: int):
    """
    features: (N, F)
    target  : (N,)
    """
    xs, ys = [], []
    N = len(target)
    for i in range(N - seq_len - horizon + 1):
        xs.append(features[i : i + seq_len])
        ys.append(target[i + seq_len + horizon - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ============ LSTMモデル ============

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.05 if num_layers > 1 else 0.0,  # ほんのりdropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]        # (batch, hidden_size)
        out = self.fc(last)         # (batch, 1)
        return out


# ============ メイン ============

def main():
    # ---- データ読み込み ----
    df = load_yf_1m_csv(CSV_FILE)
    print("Final OHLCV rows:", len(df))
    print(df.head())

    # ---- 特徴量生成 ----
    idx, feat_norm, target_norm, target_mean, target_std = build_features(df)
    print("Feature shape:", feat_norm.shape, "Target shape:", target_norm.shape)

    # ---- シーケンス化 ----
    X, y = create_sequences_multi(feat_norm, target_norm, SEQ_LEN, HORIZON)
    print("X shape:", X.shape, "y shape:", y.shape)

    if len(X) == 0:
        print("シーケンスが0件です。SEQ_LEN や HORIZON を調整してね。")
        return

    # ---- 学習/テスト分割 ----
    num_samples = len(X)
    train_size = int(num_samples * TRAIN_RATIO)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test  = X[train_size:]
    y_test  = y[train_size:]

    print("Train samples:", len(X_train), "Test samples:", len(X_test))

    # ---- Tensor変換 ----
    F = X.shape[2]  # 特徴次元
    X_train_t = torch.tensor(X_train).to(device)  # (batch, seq_len, F)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t  = torch.tensor(X_test).to(device)
    y_test_t  = torch.tensor(y_test).to(device)

    batch_size = min(BATCH_SIZE, len(X_train_t))
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ---- モデル/損失/最適化 ----
    model = LSTMModel(input_size=F, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # ---- 学習 ----
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        count = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            count += len(xb)

        if count > 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss / count:.6f}")
        else:
            print(f"[Epoch {epoch+1}/{EPOCHS}] (no batches)")

    # ---- テスト予測 ----
    model.eval()
    with torch.no_grad():
        pred_test_norm = model(X_test_t).squeeze(-1).cpu().numpy()

    # ---- 正規化を戻す ----
    y_test_denorm = y_test * target_std + target_mean
    pred_test_denorm = pred_test_norm * target_std + target_mean

    # ---- 可視化 ----
    # シーケンス+HORIZONに対応したインデックス
    # feat_norm の最初の SEQ_LEN+HORIZON-1 点までは y が無い
    full_idx = idx[SEQ_LEN + HORIZON - 1 :]
    test_idx = full_idx[train_size:]

    N_plot = min(500, len(test_idx))
    idx_plot = test_idx[-N_plot:]
    y_plot = y_test_denorm[-N_plot:]
    p_plot = pred_test_denorm[-N_plot:]

    plt.figure(figsize=(14, 6))
    plt.plot(idx_plot, y_plot, label="Actual Close", color="blue")
    plt.plot(idx_plot, p_plot, label="Predicted Close (LSTM best)", color="orange")
    plt.title(
        f"LSTM Prediction USDJPY 1m - SEQ_LEN={SEQ_LEN}, "
        f"HORIZON={HORIZON}, HIDDEN={HIDDEN_SIZE}, LAYERS={NUM_LAYERS}"
    )
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_png = "usdjpy_lstm_best.png"
    plt.savefig(out_png)
    print("Saved plot:", out_png)

    # 直近の比較も表示
    if len(y_test_denorm) > 0:
        print("\n=== Latest actual vs predicted (LSTM best) ===")
        print("Actual :", y_test_denorm[-1])
        print("Pred   :", pred_test_denorm[-1])


if __name__ == "__main__":
    main()
