# fx_transformer_usdjpy_1m.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ======== 設定 ========

CSV_FILE    = "yf_USDJPYX_1m_max.csv"  # ← ここを自分のファイル名に合わせて
SEQ_LEN     = 60            # 過去何本を入力にするか（1分足なら60=過去1時間）
TRAIN_RATIO = 0.8
EPOCHS      = 20
BATCH_SIZE  = 128
LR          = 1e-3
MAX_POINTS  = 20000         # 過去から何本使うか（多すぎると重いので上限を決める）

D_MODEL     = 64            # Transformerの特徴次元
NHEAD       = 8             # マルチヘッド数（D_MODELと割り切れる値）
NUM_LAYERS  = 3             # Encoder層数
DIM_FF      = 128           # FFNの中間次元
DROPOUT     = 0.1

# ======== デバイス判定 ========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)

# ======== データ読み込み（yfinance 1m CSV 用） ========

def load_close_prices_from_yf_1m_csv(csv_file: str) -> pd.DataFrame:
    """
    こういう形式を想定：

    Price,adj_close,close,high,low,open,volume
    Ticker,USDJPY=X,USDJPY=X,...
    Datetime,,,,,,
    2025-11-24 00:00:00+00:00,156.66,...

    → 'Price'列をdatetimeとしてIndexに、
       'close'列をfloatで取り出して DataFrame にして返す。
    """
    df = pd.read_csv(csv_file)

    # 1行目はカラム名なのでそのまま
    # 2行目 (Ticker,...) と 3行目 (Datetime,...) を削除
    # もし別形式ならここを調整
    if "Price" not in df.columns:
        raise ValueError("CSVに 'Price' 列が見当たりません。形式を確認してください。")

    # dropしないと Ticker, Datetime の行が混ざる
    df = df.drop(index=[1, 2])

    df = df.rename(columns={"Price": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")

    # close列を確保
    if "close" in df.columns:
        close_col = "close"
    elif "Close" in df.columns:
        close_col = "Close"
    else:
        raise ValueError("CSVに 'close' または 'Close' 列が見つかりません。")

    prices_df = df[[close_col]].copy()
    prices_df.columns = ["close"]
    prices_df["close"] = prices_df["close"].astype("float64")
    prices_df = prices_df.sort_index()

    print("Loaded raw rows:", len(prices_df))

    if len(prices_df) > MAX_POINTS:
        prices_df = prices_df.iloc[-MAX_POINTS:]
        print(f"Truncated to last {MAX_POINTS} points for training.")

    return prices_df

# ======== シーケンス作成 ========

def create_sequences(data: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# ======== Positional Encoding ========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# ======== Transformerモデル ========

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size=1,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,   # (batch, seq, feature) で扱う
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, 1)
        """
        # 入力を高次元に持ち上げる
        x = self.input_linear(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)   # 位置情報を付加
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        # 最後のタイムステップを使って予測
        last = x[:, -1, :]        # (batch, d_model)
        out = self.output_linear(last)  # (batch, 1)
        return out

# ======== メイン ========

def main():
    # ----- データの読み込み -----
    prices_df = load_close_prices_from_yf_1m_csv(CSV_FILE)
    print("Using rows:", len(prices_df))
    print(prices_df.head())

    prices = prices_df["close"].values.astype(np.float32)

    # 正規化
    mean = prices.mean()
    std = prices.std()
    if std == 0:
        std = 1.0
    prices_norm = (prices - mean) / std

    # シーケンス化
    X, y = create_sequences(prices_norm, SEQ_LEN)
    print("X shape:", X.shape, "y shape:", y.shape)

    if len(X) == 0:
        print("シーケンス 0 件。SEQ_LEN を小さくするか、データ量を増やしてね。")
        return

    # 学習/テスト分割
    num_samples = len(X)
    train_size = int(num_samples * TRAIN_RATIO)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test  = X[train_size:]
    y_test  = y[train_size:]

    print("Train samples:", len(X_train), "Test samples:", len(X_test))

    X_train_t = torch.tensor(X_train).unsqueeze(-1).to(device)  # (batch, seq_len, 1)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t  = torch.tensor(X_test).unsqueeze(-1).to(device)
    y_test_t  = torch.tensor(y_test).to(device)

    batch_size = min(BATCH_SIZE, len(X_train_t))
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ----- モデル・損失・最適化 -----
    model = TimeSeriesTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ----- 学習 -----
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        count = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)   # (batch,)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            count += len(xb)

        if count > 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss / count:.6f}")
        else:
            print(f"[Epoch {epoch+1}/{EPOCHS}] (no batches)")

    # ----- テストデータで予測 -----
    model.eval()
    with torch.no_grad():
        pred_test_norm = model(X_test_t).squeeze(-1).cpu().numpy()

    # 正規化を元に戻す
    y_test_denorm = y_test * std + mean
    pred_test_denorm = pred_test_norm * std + mean

    # プロット用に末尾N本だけ
    N = min(500, len(y_test_denorm))
    y_plot = y_test_denorm[-N:]
    p_plot = pred_test_denorm[-N:]

    # インデックス調整
    full_idx = prices_df.index[SEQ_LEN:]
    test_idx = full_idx[train_size:]
    idx_plot = test_idx[-N:]

    # ----- プロット -----
    plt.figure(figsize=(14, 6))
    plt.plot(idx_plot, y_plot, label="Actual Close", color="blue")
    plt.plot(idx_plot, p_plot, label="Predicted Close (Transformer)", color="orange")
    plt.title(f"Transformer Prediction USDJPY 1m - {os.path.basename(CSV_FILE)}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_png = "usdjpy_1m_transformer_prediction.png"
    plt.savefig(out_png)
    print("Saved plot:", out_png)

    if len(y_test_denorm) > 0:
        print("\n=== Latest actual vs predicted (Transformer) ===")
        print("Actual :", y_test_denorm[-1])
        print("Pred   :", pred_test_denorm[-1])


if __name__ == "__main__":
    main()
