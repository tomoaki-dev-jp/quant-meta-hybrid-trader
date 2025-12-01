import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ======== 設定 ========
CSV_FILE = "bitflyer_1m_ohlcv_history_pybitflyer.csv"  # ←ここだけ好きなCSV名に
SEQ_LEN = 30      # 過去何本から予測するか
EPOCHS = 10       # エポック数（SOROBANならもっと増やしてOK）
BATCH_SIZE = 64   # ミニバッチサイズ


# ======== デバイス判定（GPU / CPU 自動） ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======== データ読み込み ========
def load_prices(csv_file: str) -> np.ndarray:
    """
    CSVから終値(close)の配列を読み込む。
    datetime,open,high,low,close,volume 前提。
    """
    df = pd.read_csv(csv_file)

    # datetime列があれば日付として扱う（なくてもOK）
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

    if "close" not in df.columns:
        raise ValueError("CSVに 'close' 列が見つからないよ！")

    prices = df["close"].values.astype(np.float32)
    return prices


# ======== LSTM用データ作成 ========
def make_dataset(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


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
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # シーケンスの最後のステップだけ使う
        return out


def main():
    # ====== 終値読み込み ======
    prices = load_prices(CSV_FILE)
    print("Loaded prices:", len(prices), "points")

    if len(prices) <= SEQ_LEN + 5:
        print("データが少なすぎて学習できないかも… SEQ_LEN を小さくするか、データを増やしてね。")
    
    # 正規化（平均0・分散1）
    mean = prices.mean()
    std = prices.std() if prices.std() > 0 else 1.0
    prices_norm = (prices - mean) / std

    # LSTM用データ作成
    X, y = make_dataset(prices_norm, SEQ_LEN)
    print("X shape:", X.shape, "y shape:", y.shape)

    if len(X) == 0:
        print("学習データが 0 件なので学習をスキップします。")
        return

    X = torch.tensor(X).unsqueeze(-1).to(device)  # (batch, seq, 1)
    y = torch.tensor(y).to(device)

    # ====== モデル構築 ======
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # バッチサイズ調整（データが少ないとき用）
    batch_size = min(BATCH_SIZE, len(X))

    # ====== 学習 ======
    for epoch in range(EPOCHS):
        # シャッフル
        perm = torch.randperm(len(X))
        X_epoch = X[perm]
        y_epoch = y[perm]

        loss = None

        for i in range(0, len(X), batch_size):
            xb = X_epoch[i:i + batch_size]
            yb = y_epoch[i:i + batch_size]

            if len(xb) == 0:
                continue

            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        if loss is not None:
            print(f"[Epoch {epoch+1}] Loss: {loss.item():.6f}")
        else:
            print(f"[Epoch {epoch+1}] (no training batches)")

    # ====== 未来1ステップ予測 ======
    recent = prices_norm[-SEQ_LEN:]  # 最新のシーケンス
    inp = torch.tensor(recent).unsqueeze(0).unsqueeze(-1).to(device)

    model.eval()
    with torch.no_grad():
        pred_norm = model(inp).item()

    pred_price = pred_norm * std + mean

    print("\n=== BTC/JPY 未来1ステップ（1分 or 任意の足）予測値 ===")
    print(f"予測価格: {pred_price:,.0f} 円")


if __name__ == "__main__":
    main()
