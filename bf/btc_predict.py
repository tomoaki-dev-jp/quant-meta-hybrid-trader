import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ================================
#   デバイス判定（GPU / CPU 自動）
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================
#   データ読み込み
# ================================
df = pd.read_csv("/usr/src/app/bitflyer_1m_ohlcv_history.csv")

# 終値だけ使う
prices = df["close"].values.astype(np.float32)

# スケーリング（簡易版）
mean = prices.mean()
std = prices.std()
prices_norm = (prices - mean) / std

# ================================
#   LSTM用データ作成
# ================================
SEQ_LEN = 30  # 過去30分から未来1分を予測

def make_dataset(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = make_dataset(prices_norm, SEQ_LEN)

X = torch.tensor(X).unsqueeze(-1).to(device)  # (batch, seq, 1)
y = torch.tensor(y).to(device)

# ================================
#   LSTMモデル定義
# ================================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================
#   トレーニング
# ================================
EPOCHS = 5  # お試し用（SOROBANでは増やせる）
BATCH = 64

for epoch in range(EPOCHS):
    perm = torch.randperm(len(X))
    X_epoch = X[perm]
    y_epoch = y[perm]

    loss = None  # ← 追加：初期化

    for i in range(0, len(X), BATCH):
        xb = X_epoch[i:i+BATCH]
        yb = y_epoch[i:i+BATCH]

        if len(xb) == 0:
            continue  # 念のため安全対策

        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    # ← loss が1度も定義されなかった場合の安全ガード
    if loss is not None:
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.6f}")
    else:
        print(f"[Epoch {epoch+1}] (no training batches)")

# ================================
#   未来1ステップを予測
# ================================
recent = prices_norm[-SEQ_LEN:]
inp = torch.tensor(recent).unsqueeze(0).unsqueeze(-1).to(device)

model.eval()
with torch.no_grad():
    pred_norm = model(inp).item()

# 元スケールに戻す
pred_price = pred_norm * std + mean

print("\n=== BTC/JPY 未来1分の予測値 ===")
print(f"予測価格: {pred_price:,.0f} 円")
