# fx_transformer_usdjpy_walkforward.py
#
# USDJPY 5m の yfinance CSV を読み込み、
# Transformer で1ステップ先のcloseを予測。
# ・LSTM版と同じリッチ特徴量
# ・通常のtrain/test評価
# ・シンプルなウォークフォワード検証つき

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ========= 設定 =========

CSV_FILE      = "fx/yf_USDJPYX_5m_max.csv"
USE_RESAMPLE  = False
RESAMPLE_RULE = "5min"

SEQ_LEN      = 96      # 過去8時間
HORIZON      = 1       # 1本先予測
TRAIN_RATIO  = 0.8
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 3e-4
MAX_POINTS   = 20000

D_MODEL      = 128
NHEAD        = 8
NUM_LAYERS   = 3
DIM_FF       = 256
DROPOUT      = 0.2

EARLY_STOP_PATIENCE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)

# ========= CSV読み込み =========

def load_yf_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    if "Price" not in df.columns:
        raise ValueError("CSVに 'Price' 列が見当たりません。")

    mask_bad = df["Price"].astype(str).str.contains("Ticker|Datetime", na=False)
    df = df[~mask_bad].copy()

    df = df.rename(columns={"Price": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSVに {col} がありません。")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

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

    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]

    print("Loaded rows:", len(df))
    return df

# ========= 特徴量作成（LSTM精度爆伸び版と同じノリ） =========

def build_features(df: pd.DataFrame):
    feat_df = pd.DataFrame(index=df.index)
    close = df["close"]
    open_ = df["open"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # 基本系
    feat_df["close"] = close
    feat_df["ret1"] = close.pct_change().fillna(0.0)
    feat_df["hl_spread"] = (high - low) / (close + 1e-9)
    feat_df["oc_change"] = (close - open_) / (open_ + 1e-9)

    # ボラ系
    feat_df["ret_std_1h"] = close.pct_change().rolling(12).std()
    feat_df["ret_std_3h"] = close.pct_change().rolling(36).std()

    # MA系
    ma_fast = close.rolling(12).mean()
    ma_slow = close.rolling(36).mean()
    feat_df["ma_fast"] = ma_fast
    feat_df["ma_slow"] = ma_slow
    feat_df["ma_ratio"] = (ma_fast - ma_slow) / (ma_slow + 1e-9)

    # ボリューム系
    feat_df["volume"] = vol
    vol_ma = vol.rolling(48).mean()
    vol_std = vol.rolling(48).std()
    feat_df["vol_zscore"] = (vol - vol_ma) / (vol_std + 1e-9)

    # 時間特徴
    idx = df.index
    hour = idx.hour
    dow = idx.dayofweek
    feat_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat_df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    feat_df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

    # ローリング実現ボラ
    feat_df["vol_roll"] = close.pct_change().rolling(30).std()

    # NaN/inf除去
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

    target = feat_df["close"].values.astype(np.float32)
    target_mean = target.mean()
    target_std  = target.std() if target.std() > 0 else 1.0
    target_norm = (target - target_mean) / target_std

    feat_values = feat_df.values.astype(np.float32)
    feat_mean = feat_values.mean(axis=0, keepdims=True)
    feat_std = feat_values.std(axis=0, keepdims=True)
    feat_std[feat_std == 0] = 1.0
    feat_norm = (feat_values - feat_mean) / feat_std

    return feat_df.index, feat_norm, target_norm, target_mean, target_std

# ========= シーケンス =========

def create_sequences(features, target, seq_len, horizon):
    xs, ys = [], []
    N = len(target)
    for i in range(N - seq_len - horizon + 1):
        xs.append(features[i:i+seq_len])
        ys.append(target[i + seq_len + horizon - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# ========= Positional Encoding =========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        return x + self.pe[:seq_len].to(x.device)

# ========= Transformerモデル =========

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)              # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)             # (seq_len, batch, d_model)
        x = self.pos_enc(x)
        out = self.encoder(x)              # (seq_len, batch, d_model)
        last = out[-1]                     # (batch, d_model)
        return self.fc(last).squeeze(-1)   # (batch,)

# ========= 未来予測 =========

def forecast_future(model, feat_norm, target_mean, target_std, seq_len, steps=50):
    model.eval()
    seq = feat_norm[-seq_len:]
    seq_t = torch.tensor(seq).unsqueeze(0).to(device)  # (1, seq_len, F)
    preds = []

    with torch.no_grad():
        for _ in range(steps):
            pred_norm = model(seq_t).item()
            pred_price = pred_norm * target_std + target_mean
            preds.append(pred_price)

            # close 成分だけ更新（0番目と仮定）
            last_feat = seq_t[0, -1, :].clone()
            last_feat[0] = pred_norm
            new_seq = torch.cat([seq_t[0, 1:, :], last_feat.unsqueeze(0)], dim=0)
            seq_t = new_seq.unsqueeze(0)

    return np.array(preds)

# ========= 学習（通常） =========

def train_transformer(X_train, y_train, X_val, y_val):
    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_val_t   = torch.tensor(X_val).to(device)
    y_val_t   = torch.tensor(y_val).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=min(BATCH_SIZE, len(X_train_t)),
        shuffle=True,
    )

    model = TimeSeriesTransformer(
        input_size=X_train.shape[2],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.HuberLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_state = None
    best_val = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        loss_sum = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            total += len(xb)
        train_loss = loss_sum / max(total, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train {train_loss:.6f}  Val {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# ========= ウォークフォワード検証 =========

def walk_forward_eval(X, y, idx, target_mean, target_std,
                      train_window=5000, test_window=1000):
    """
    時系列を時間順のまま
    [train_window] で学習 → 直後の [test_window] で評価
    を繰り返す簡易ウォークフォワード。
    """
    N = len(X)
    results = []

    start = 0
    k = 1
    while True:
        train_start = start
        train_end   = train_start + train_window
        test_end    = train_end + test_window
        if test_end > N:
            break

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test  = X[train_end:test_end]
        y_test  = y[train_end:test_end]

        print(f"\n=== WF Fold {k} ===")
        print(f"Train: {idx[train_start]} ~ {idx[train_end-1]}")
        print(f"Test : {idx[train_end]} ~ {idx[test_end-1]}")

        model = train_transformer(X_train, y_train, X_test, y_test)

        with torch.no_grad():
            X_test_t = torch.tensor(X_test).to(device)
            pred_norm = model(X_test_t).cpu().numpy()

        y_den = y_test * target_std + target_mean
        p_den = pred_norm * target_std + target_mean

        mse = np.mean((p_den - y_den) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(p_den - y_den))
        mape = np.mean(np.abs((p_den - y_den) / y_den)) * 100

        print(f"Fold {k} RMSE={rmse:.5f} MAE={mae:.5f} MAPE={mape:.3f}%")
        results.append((rmse, mae, mape))

        k += 1
        start += test_window   # テスト窓分だけ前進

    if results:
        rmses, maes, mapes = zip(*results)
        print("\n=== Walk-forward summary ===")
        print(f"Avg RMSE: {np.mean(rmses):.5f}")
        print(f"Avg MAE : {np.mean(maes):.5f}")
        print(f"Avg MAPE: {np.mean(mapes):.3f}%")
    else:
        print("Not enough data for walk-forward eval.")

# ========= メイン =========

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

    # 通常学習
    print("\n==== Train/Test (single split) ====")
    model = train_transformer(X_train, y_train, X_test, y_test)

    # テストセットで評価
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).to(device)
        pred_norm = model(X_test_t).cpu().numpy()

    y_test_den = y_test * target_std + target_mean
    pred_den   = pred_norm * target_std + target_mean

    full_idx = idx[SEQ_LEN + HORIZON - 1:]
    test_idx = full_idx[train_size:]

    mse = np.mean((pred_den - y_test_den) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_den - y_test_den))
    mape = np.mean(np.abs((pred_den - y_test_den) / y_test_den)) * 100

    print("\n=== Test Metrics (single split) ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MAPE: {mape:.4f}%")

    N_plot = min(500, len(test_idx))
    plt.figure(figsize=(14, 6))
    plt.plot(test_idx[-N_plot:], y_test_den[-N_plot:], label="Actual")
    plt.plot(test_idx[-N_plot:], pred_den[-N_plot:], label="Predicted")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("transformer_pred.png")
    print("Saved: transformer_pred.png")

    # 未来予測
    future_steps = 100
    future_preds = forecast_future(
        model, feat_norm, target_mean, target_std,
        seq_len=SEQ_LEN, steps=future_steps
    )
    last_time = idx[-1]
    future_times = [
        last_time + pd.Timedelta(minutes=5 * i) for i in range(1, future_steps + 1)
    ]

    plt.figure(figsize=(14, 6))
    plt.plot(idx[-500:], df["close"].values[-500:], label="Actual")
    plt.plot(test_idx[-N_plot:], pred_den[-N_plot:], label="Predicted")
    plt.plot(future_times, future_preds, label="Future Forecast")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("transformer_pred_with_future.png")
    print("Saved: transformer_pred_with_future.png")

    # ウォークフォワード
    print("\n==== Walk-forward evaluation ====")
    walk_forward_eval(
        X, y, idx[SEQ_LEN + HORIZON - 1:], target_mean, target_std,
        train_window=5000, test_window=1000
    )


if __name__ == "__main__":
    main()
