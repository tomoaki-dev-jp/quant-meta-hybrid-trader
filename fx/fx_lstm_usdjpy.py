# fx_lstm_usdjpy_1m.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ======== 設定 ========

CSV_FILE    = "yf_USDJPYX_1m_max.csv"  # あなたのCSVファイル名
SEQ_LEN     = 60            # 過去何本で予測するか（1分足なら60=過去1時間分）
TRAIN_RATIO = 0.8           # 学習データの割合
EPOCHS      = 100            # エポック数（GPUならもっと増やしてもOK）
BATCH_SIZE  = 128           # ミニバッチサイズ
LR          = 1e-3          # 学習率
MAX_POINTS  = 20000         # 過去から何本分だけ使うか（多すぎると重いので絞る）

# ======== デバイス判定（GPU / CPU 自動） ========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)

# ======== データ読み込み ========

def load_close_prices_from_yf_1m_csv(csv_file: str) -> pd.DataFrame:
    """
    yfinance 形式:
    Price,adj_close,close,high,low,open,volume
    Ticker,USDJPY=X,USDJPY=X,...
    Datetime,,,,,,
    2025-11-24 00:00:00+00:00, ...
    ...
    を想定して、終値 'close' と datetime を整形して返す。
    """
    df = pd.read_csv(csv_file)

    # 2行目(Ticker...)と3行目(Datetime...)を削除
    # → 1行目がcol名、2行目/3行目が邪魔な行なので落とす
    # Index 0: header, 1: Ticker..., 2: Datetime..., 3以降がデータ
    df = df.drop(index=[0, 1]) if "Price" in df.columns else df

    # 列名が "Price, adj_close, close, high, low, open, volume" のはず
    # Price列を datetime として扱う
    if "Price" not in df.columns:
        raise ValueError("CSVに 'Price' 列が見つかりません。形式を確認してください。")

    df = df.rename(columns={"Price": "datetime"})

    # datetimeに変換
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")

    # 必要なカラムだけ抽出
    # close列を小文字に揃える
    if "close" in df.columns:
        close_col = "close"
    elif "Close" in df.columns:
        close_col = "Close"
    else:
        raise ValueError("CSVに 'close' または 'Close' 列が見つかりません。")

    prices_df = df[[close_col]].copy()
    prices_df.columns = ["close"]

    # 型をfloatに
    prices_df["close"] = prices_df["close"].astype("float64")
    prices_df = prices_df.sort_index()

    print("Loaded raw rows:", len(prices_df))

    # 最新の MAX_POINTS 本だけ使う（データが多すぎると重いので）
    if len(prices_df) > MAX_POINTS:
        prices_df = prices_df.iloc[-MAX_POINTS:]
        print(f"Truncated to last {MAX_POINTS} points for training.")

    return prices_df


# ======== LSTM用データセット作成 ========

def create_sequences(data: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ======== LSTMモデル ========

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden)
        out = self.fc(last)   # (batch, 1)
        return out


def main():
    # ===== データ読み込み =====
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

    # シーケンス作成
    X, y = create_sequences(prices_norm, SEQ_LEN)
    print("X shape:", X.shape, "y shape:", y.shape)

    if len(X) == 0:
        print("シーケンスが0件です。SEQ_LENを小さくするかデータを増やしてください。")
        return

    # 学習 / テスト分割
    num_samples = len(X)
    train_size = int(num_samples * TRAIN_RATIO)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test  = X[train_size:]
    y_test  = y[train_size:]

    print("Train samples:", len(X_train), "Test samples:", len(X_test))

    # Tensor変換
    X_train_t = torch.tensor(X_train).unsqueeze(-1).to(device)  # (batch, seq_len, 1)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t  = torch.tensor(X_test).unsqueeze(-1).to(device)
    y_test_t  = torch.tensor(y_test).to(device)

    # DataLoader
    batch_size = min(BATCH_SIZE, len(X_train_t))
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # モデル & 損失 & オプティマイザ
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ===== 学習 =====
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        count = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)  # (batch,)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            count += len(xb)

        if count > 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss / count:.6f}")
        else:
            print(f"[Epoch {epoch+1}/{EPOCHS}] (no batches)")

    # ===== テストデータで予測 =====
    model.eval()
    with torch.no_grad():
        pred_test_norm = model(X_test_t).squeeze(-1).cpu().numpy()

    # 正規化を戻す
    y_test_denorm = y_test * std + mean
    pred_test_denorm = pred_test_norm * std + mean

    # プロット用に末尾N本だけ
    N = min(500, len(y_test_denorm))  # 最新500本だけ描画
    y_plot = y_test_denorm[-N:]
    p_plot = pred_test_denorm[-N:]

    # 日付Indexを合わせる
    full_idx = prices_df.index[SEQ_LEN:]      # シーケンスのyに対応するインデックス
    test_idx = full_idx[train_size:]         # テスト部分
    idx_plot = test_idx[-N:]

    # ===== プロット =====
    plt.figure(figsize=(14, 6))
    plt.plot(idx_plot, y_plot, label="Actual Close", color="blue")
    plt.plot(idx_plot, p_plot, label="Predicted Close", color="orange")
    plt.title(f"LSTM Prediction USDJPY 1m - {os.path.basename(CSV_FILE)}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_png = "usdjpy_1m_lstm_prediction.png"
    plt.savefig(out_png)
    print("Saved plot:", out_png)

    # 直近の1本の比較表示
    if len(y_test_denorm) > 0:
        print("\n=== Latest actual vs predicted ===")
        print("Actual :", y_test_denorm[-1])
        print("Pred   :", pred_test_denorm[-1])


if __name__ == "__main__":
    main()
