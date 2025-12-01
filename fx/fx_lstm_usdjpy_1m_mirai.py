# fx_lstm_usdjpy_5m_best_with_forecast.py
#
# yfinance の CSV（1m/5m問わず）を読み込み、
# 多特徴量LSTMにより未来を予測する「最強構成」＋未来10ステップ予測付き。

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============ 設定 ============

CSV_FILE     = "fx/yf_USDJPYX_5m_max.csv"
USE_RESAMPLE = False                    # Trueならリサンプリング可能
RESAMPLE_RULE = "5min"

SEQ_LEN      = 72       # 5分足なら過去6時間
HORIZON      = 1        # 1本先予測（5分後）
TRAIN_RATIO  = 0.8
EPOCHS       = 300
BATCH_SIZE   = 128
LR           = 1e-4
MAX_POINTS   = 20000

HIDDEN_SIZE  = 128
NUM_LAYERS   = 3

# ============ デバイス ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)


# ============ CSV読み込み ============
def load_yf_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    if "Price" not in df.columns:
        raise ValueError("CSVに 'Price' 列が見当たりません。")

    # Ticker/Datetime行除外
    mask_bad = df["Price"].astype(str).str.contains("Ticker|Datetime", na=False)
    df = df[~mask_bad].copy()

    df = df.rename(columns={"Price": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSVに {col} がありません。")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    print("Loaded rows:", len(df))

    if USE_RESAMPLE:
        df = df[["open", "high", "low", "close", "volume"]].resample(
            RESAMPLE_RULE
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        print("Resampled:", len(df))

    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]
        print(f"Trimmed to last {MAX_POINTS} rows")

    return df


# ============ 特徴量 ============
def build_features(df: pd.DataFrame):
    feat_df = pd.DataFrame(index=df.index)

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low  = df["low"]
    vol  = df["volume"]

    feat_df["close"] = close
    feat_df["ret1"] = close.pct_change().fillna(0.0)
    feat_df["hl_spread"] = (high - low) / (close + 1e-9)
    feat_df["oc_change"] = (close - open_) / (open_ + 1e-9)
    feat_df["volume"] = vol
    feat_df["vol_roll"] = close.pct_change().rolling(30).std().fillna(0.0)

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

    target = feat_df["close"].values.astype(np.float32)
    target_mean = target.mean()
    target_std  = target.std() if target.std() > 0 else 1.0
    target_norm = (target - target_mean) / target_std

    # 特徴量を標準化
    feat_values = feat_df.values.astype(np.float32)
    feat_mean = feat_values.mean(axis=0, keepdims=True)
    feat_std  = feat_values.std(axis=0, keepdims=True)
    feat_std[feat_std == 0] = 1.0
    feat_norm = (feat_values - feat_mean) / feat_std

    return feat_df.index, feat_norm, target_norm, target_mean, target_std


# ============ シーケンス作成 ============
def create_sequences(features, target, seq_len, horizon):
    xs, ys = [], []
    N = len(target)
    for i in range(N - seq_len - horizon + 1):
        xs.append(features[i:i+seq_len])
        ys.append(target[i + seq_len + horizon - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ============ LSTMモデル ============
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.05 if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


# ============ 未来予測（再帰的） ============
def forecast_future(model, feat_norm, target_mean, target_std, steps=10):
    model.eval()

    seq = feat_norm[-SEQ_LEN:]  # (SEQ_LEN, F)
    seq_t = torch.tensor(seq).unsqueeze(0).to(device)

    preds = []

    with torch.no_grad():
        for _ in range(steps):
            pred_norm = model(seq_t).item()
            pred_price = pred_norm * target_std + target_mean
            preds.append(pred_price)

            last_feat = seq_t[0, -1, :].clone()
            last_feat[0] = pred_norm

            new_seq = torch.cat([seq_t[0, 1:, :], last_feat.unsqueeze(0)], dim=0)
            seq_t = new_seq.unsqueeze(0)

    return np.array(preds)


# ============ メイン ============
def main():
    df = load_yf_csv(CSV_FILE)
    idx, feat_norm, target_norm, target_mean, target_std = build_features(df)

    X, y = create_sequences(feat_norm, target_norm, SEQ_LEN, HORIZON)
    print("X:", X.shape, "y:", y.shape)

    N = len(X)
    train_size = int(N * TRAIN_RATIO)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test  = X[train_size:]
    y_test  = y[train_size:]

    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t  = torch.tensor(X_test).to(device)
    y_test_t  = torch.tensor(y_test).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=min(BATCH_SIZE, len(X_train_t)),
        shuffle=True,
    )

    model = LSTMModel(
        input_size=X.shape[2],
        hidden=HIDDEN_SIZE,
        layers=NUM_LAYERS
    ).to(device)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # ---- 学習 ----
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        loss_sum = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(xb)
            total += len(xb)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {loss_sum/total:.6f}")

    # ---- テスト予測 ----
    model.eval()
    pred_test_norm = model(X_test_t).squeeze(-1).detach().cpu().numpy()

    y_test_denorm = y_test * target_std + target_mean
    pred_test_denorm = pred_test_norm * target_std + target_mean

    # ---- グラフ ----
    full_idx = idx[SEQ_LEN + HORIZON - 1:]
    test_idx = full_idx[train_size:]

    N_plot = min(500, len(test_idx))
    plt.figure(figsize=(14, 6))
    plt.plot(test_idx[-N_plot:], y_test_denorm[-N_plot:], label="Actual")
    plt.plot(test_idx[-N_plot:], pred_test_denorm[-N_plot:], label="Predicted")
    plt.legend()
    plt.grid()
    plt.savefig("lstm_pred.png")
    print("Saved: lstm_pred.png")

    # ---- 未来予測 ----
    future = forecast_future(model, feat_norm, target_mean, target_std, steps=10)

    print("\n=== Future Forecast (next steps) ===")
    for i, p in enumerate(future, 1):
        print(f"+{i} : {p:.3f}")


if __name__ == "__main__":
    main()
