import torch

# ここで自動判定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# 例：テンソル作るとき
x = torch.randn(32, 10, device=device)
w = torch.randn(10, 1, device=device)

y = x @ w  # ふつうに計算
