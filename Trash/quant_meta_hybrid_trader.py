# quant_meta_hybrid_trader.py
#
# 全部乗せ研究用トレーダーフレームワーク v2
#
# ★入ってる要素★
# 1. マルチアセット（複数通貨ペア）
# 2. LSTM & Transformer で +1,+3,+6,+12,+24 リターン予測
# 3. Regime CNN で Trend / Range / HighVol を分類
# 4. ForecastFusionNet で LSTM + Transformer + Regime を統合
# 5. 強化報酬 + 実運用コスト（スプレッド/スリッページ） + 7段階ロット + PPO
# 6. ランダムサーチによる簡易メタサーチ（lr, loss_factor, trend_boost）
# 7. 自己教師ありプリトレ(LSTMマスク復元)
# 8. ノイズ除去（EMAフィルタ）を特徴量に追加
# 9. OrderBookフック（今はダミーだが後から本物を刺せるようにしてある）
# 10. 推論モードシミュレーションで行動ログ＆EQカーブ出力
#
# ※ 完全に「遊び・研究」用。実運用で使うのは絶対NG！

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ================== 設定 ==================

PAIR_CSV_LIST = [
    "yf_USDJPYX_5m_max.csv",
    # ここに他の通貨ペアのCSVを足してOK
    # "fx/yf_EURUSD_5m_max.csv",
    # "fx/yf_GBPUSD_5m_max.csv",
]

USE_RESAMPLE  = False
RESAMPLE_RULE = "5min"
MAX_POINTS    = 20000

STATE_RET_LEN = 48

FORECAST_HORIZONS = [1, 3, 6, 12, 24]
N_HORIZON = len(FORECAST_HORIZONS)

LSTM_SEQ_LEN  = 64
LSTM_EPOCHS   = 15
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
LSTM_LR       = 1e-3
LSTM_BATCH    = 256

TF_D_MODEL    = 64
TF_NHEAD      = 4
TF_LAYERS     = 2
TF_FF         = 128
TF_EPOCHS     = 15
TF_LR         = 1e-3
TF_BATCH      = 256

REGIME_SEQ_LEN = 64
REGIME_EPOCHS  = 10
REGIME_LR      = 1e-3
REGIME_BATCH   = 256

EPISODES_PER_PAIR = 20
STEPS_PER_EP      = 1200

GAMMA       = 0.99
LAMBDA_GAE  = 0.95
CLIP_EPS    = 0.2
EPOCHS_PPO  = 4
MINI_BATCH  = 1024

N_ACTIONS    = 7  # [-5,-3,-1,0,1,3,5]
MAX_POSITION = 5

# ---- 元々のボリュームコスト ----
TRANSACTION_COST   = 0.00003
LOSS_FACTOR        = 1.2
TREND_THRESHOLD    = 0.0001
TREND_BOOST        = 2.0
LSTM_REWARD_SCALE  = 0.3
TF_REWARD_SCALE    = 0.3

# ---- 実運用コスト系（USDJPY想定）----
SPREAD_PIPS    = 0.02   # 往復2pips相当
SLIPPAGE_PIPS  = 0.01   # 片道1pips相当
PIP_VALUE_JPY  = 0.01   # 1pip=0.01円（ざっくり）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print("Using device:", device)

# ================== ユーティリティ ==================

def ema_filter(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """単純な指数移動平均フィルタ"""
    y = np.zeros_like(x, dtype=np.float32)
    if len(x) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def load_close_series(csv_file: str) -> pd.DataFrame:
    print(f"[load_close_series] loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    if "Price" not in df.columns:
        raise ValueError("CSVに 'Price' 列が見当たりません。")

    mask_bad = df["Price"].astype(str).str.contains("Ticker|Datetime", na=False)
    df = df[~mask_bad].copy()

    df = df.rename(columns={"Price": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    if "close" not in df.columns:
        raise ValueError("CSVに close 列が必要です。")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    if USE_RESAMPLE:
        df = df[["close"]].resample(RESAMPLE_RULE).last().dropna()

    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]
        print(f"[load_close_series] trimmed to last {MAX_POINTS} rows")

    print("[load_close_series] final rows:", len(df))
    return df


def build_returns_and_tech(prices: np.ndarray):
    """
    returns, vol_12, vol_36, trend_36, rsi, returns_smooth を返す
    """
    returns = np.diff(prices) / prices[:-1]
    r_series = pd.Series(returns)

    vol_12   = r_series.rolling(12).std().fillna(0.0).values
    vol_36   = r_series.rolling(36).std().fillna(0.0).values
    trend_36 = r_series.rolling(36).mean().fillna(0.0).values

    up = r_series.clip(lower=0).rolling(14).mean()
    down = (-r_series.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 * up / (up + down + 1e-9)
    rsi = ((rsi - 50.0) / 50.0).fillna(0.0).values

    returns_smooth = ema_filter(returns, alpha=0.1)

    return (
        returns.astype(np.float32),
        vol_12.astype(np.float32),
        vol_36.astype(np.float32),
        trend_36.astype(np.float32),
        rsi.astype(np.float32),
        returns_smooth.astype(np.float32),
    )


def regime_label_from_threshold(trend: float, vol: float) -> int:
    """0: Range, 1: Trend, 2: HighVol の簡易ラベル"""
    if abs(trend) > TREND_THRESHOLD and vol < 3 * TREND_THRESHOLD:
        return 1
    elif vol > 3 * TREND_THRESHOLD:
        return 2
    else:
        return 0

# ================== Dataset 作成 ==================

def build_forecast_dataset(prices: np.ndarray):
    # LSTM/Transformer用
    returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)

    feat_mat = np.stack(
        [returns, vol_12, vol_36, trend_36, rsi, returns_smooth],
        axis=1
    )  # (T-1, 6)

    r = returns
    T = feat_mat.shape[0]
    seq_len = LSTM_SEQ_LEN
    horizon_max = max(FORECAST_HORIZONS)

    X_list, y_list = [], []
    for t in range(seq_len, T - horizon_max):
        X_list.append(feat_mat[t - seq_len: t])
        targets = []
        for h in FORECAST_HORIZONS:
            targets.append(r[t + h - 1])
        y_list.append(targets)

    X_seq = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print("[build_forecast_dataset] X_seq:", X_seq.shape, "y:", y.shape)
    return X_seq, y


def build_regime_dataset(prices: np.ndarray):
    returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)
    r = returns
    seq_len = REGIME_SEQ_LEN
    X_list, y_list = [], []

    for t in range(seq_len, len(r)):
        window = r[t - seq_len: t]
        vol = vol_36[t]
        trend = trend_36[t]
        label = regime_label_from_threshold(trend, vol)
        X_list.append(window)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print("[build_regime_dataset] X:", X.shape, "y:", y.shape)
    return X, y

# ================== モデル群 ==================

class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, N_HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(1)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len].to(x.device)


class MultiHorizonTransformer(nn.Module):
    def __init__(
        self,
        input_dim=6,
        d_model=TF_D_MODEL,
        nhead=TF_NHEAD,
        num_layers=TF_LAYERS,
        dim_ff=TF_FF,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, N_HORIZON)

    def forward(self, x):
        x = self.input_proj(x)      # (B,L,d)
        x = x.permute(1, 0, 2)     # (L,B,d)
        x = self.pos_enc(x)
        out = self.encoder(x)
        last = out[-1]
        return self.fc(last)


class RegimeCNN(nn.Module):
    def __init__(self, in_len=REGIME_SEQ_LEN, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)         # (B,1,L)
        h = self.net(x).squeeze(-1)  # (B,32)
        return self.fc(h)


class ForecastFusionNet(nn.Module):
    """LSTM予測 + Transformer予測 + Regime(one-hot) → 統合予測"""
    def __init__(self):
        super().__init__()
        in_dim = N_HORIZON * 2 + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, N_HORIZON),
        )

    def forward(self, lstm_pred, tf_pred, regime_onehot):
        x = torch.cat([lstm_pred, tf_pred, regime_onehot], dim=-1)
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head  = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value  = self.value_head(h)
        return logits, value.squeeze(-1)

# ================== 自己教師ありプリトレ（LSTM） ==================

def pretrain_lstm_self_supervised(prices: np.ndarray, epochs: int = 5):
    """
    リターン系列に対してランダムマスク復元をさせる簡易プリトレ。
    ForecasterのLSTMの初期重みに使う。
    """
    returns, *_ = build_returns_and_tech(prices)
    r = returns
    seq_len = LSTM_SEQ_LEN
    X_list, y_list = [], []

    for t in range(seq_len, len(r)):
        window = r[t-seq_len:t]
        mask_idx = np.random.randint(0, seq_len)
        target_val = window[mask_idx]
        window_masked = window.copy()
        window_masked[mask_idx] = 0.0
        X_list.append(window_masked)
        y_list.append(target_val)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(X, device=device),
        torch.tensor(y, device=device)
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=min(512, len(ds)), shuffle=True
    )

    lstm = nn.LSTM(
        input_size=1,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        batch_first=True,
        dropout=0.1 if LSTM_LAYERS > 1 else 0.0,
    ).to(device)
    fc = nn.Linear(LSTM_HIDDEN, 1).to(device)

    params = list(lstm.parameters()) + list(fc.parameters())
    opt = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()

    for ep in range(1, epochs+1):
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader:
            xb = xb.unsqueeze(-1)  # (B,L,1)
            opt.zero_grad()
            out, _ = lstm(xb)
            last = out[:, -1, :]
            pred = fc(last).squeeze(-1)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        print(f"[Pretrain LSTM] Epoch {ep}/{epochs} "
              f"Loss={loss_sum/max(tot,1):.6e}")

    return lstm.state_dict()

# ================== Forecaster & Regime 学習 ==================

def train_forecasters_and_regime(prices: np.ndarray):
    # --- 自己教師ありプリトレ ---
    print("[Pretrain] LSTM self-supervised pretraining...")
    pretrained_lstm_state = pretrain_lstm_self_supervised(prices, epochs=5)

    # --- forecaster dataset ---
    X_seq, y = build_forecast_dataset(prices)
    N = len(X_seq)
    train_size = int(N * 0.8)
    X_train = torch.tensor(X_seq[:train_size], device=device)
    y_train = torch.tensor(y[:train_size], device=device)
    X_val   = torch.tensor(X_seq[train_size:], device=device)
    y_val   = torch.tensor(y[train_size:], device=device)

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    loader_lstm = torch.utils.data.DataLoader(
        ds_train,
        batch_size=min(LSTM_BATCH, len(ds_train)),
        shuffle=True,
    )
    loader_tf = torch.utils.data.DataLoader(
        ds_train,
        batch_size=min(TF_BATCH, len(ds_train)),
        shuffle=True,
    )

    crit = nn.MSELoss()

    # --- LSTM ---
    lstm_model = MultiHorizonLSTM(input_dim=X_seq.shape[2]).to(device)
    # LSTM部分だけプリトレ重みを流用（形が合うパラメータのみ）
    with torch.no_grad():
        lstm_state = lstm_model.lstm.state_dict()
        for name, p in lstm_state.items():
            if name in pretrained_lstm_state and \
               pretrained_lstm_state[name].shape == p.shape:
                p.copy_(pretrained_lstm_state[name])
        lstm_model.lstm.load_state_dict(lstm_state)

    opt_lstm = optim.AdamW(lstm_model.parameters(), lr=LSTM_LR, weight_decay=1e-4)

    for epoch in range(1, LSTM_EPOCHS + 1):
        lstm_model.train()
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader_lstm:
            opt_lstm.zero_grad()
            pred = lstm_model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt_lstm.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        train_loss = loss_sum / max(tot, 1)
        lstm_model.eval()
        with torch.no_grad():
            val_pred = lstm_model(X_val)
            val_loss = crit(val_pred, y_val).item()
        print(f"[LSTM] Epoch {epoch}/{LSTM_EPOCHS} "
              f"Train={train_loss:.6e} Val={val_loss:.6e}")

    # --- Transformer ---
    tf_model = MultiHorizonTransformer(input_dim=X_seq.shape[2]).to(device)
    opt_tf = optim.AdamW(tf_model.parameters(), lr=TF_LR, weight_decay=1e-4)

    for epoch in range(1, TF_EPOCHS + 1):
        tf_model.train()
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader_tf:
            opt_tf.zero_grad()
            pred = tf_model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt_tf.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        train_loss = loss_sum / max(tot, 1)
        tf_model.eval()
        with torch.no_grad():
            val_pred = tf_model(X_val)
            val_loss = crit(val_pred, y_val).item()
        print(f"[Transformer] Epoch {epoch}/{TF_EPOCHS} "
              f"Train={train_loss:.6e} Val={val_loss:.6e}")

    # --- Regime CNN ---
    X_reg, y_reg = build_regime_dataset(prices)
    N_reg = len(X_reg)
    train_size_reg = int(N_reg * 0.8)
    Xr_train = torch.tensor(X_reg[:train_size_reg], device=device)
    yr_train = torch.tensor(y_reg[:train_size_reg], device=device)
    Xr_val   = torch.tensor(X_reg[train_size_reg:], device=device)
    yr_val   = torch.tensor(y_reg[train_size_reg:], device=device)

    ds_reg = torch.utils.data.TensorDataset(Xr_train, yr_train)
    loader_reg = torch.utils.data.DataLoader(
        ds_reg,
        batch_size=min(REGIME_BATCH, len(ds_reg)),
        shuffle=True,
    )

    regime_model = RegimeCNN().to(device)
    opt_reg = optim.AdamW(regime_model.parameters(), lr=REGIME_LR)
    crit_reg = nn.CrossEntropyLoss()

    for epoch in range(1, REGIME_EPOCHS + 1):
        regime_model.train()
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader_reg:
            opt_reg.zero_grad()
            logits = regime_model(xb)
            loss = crit_reg(logits, yb)
            loss.backward()
            opt_reg.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        train_loss = loss_sum / max(tot, 1)
        regime_model.eval()
        with torch.no_grad():
            logits_val = regime_model(Xr_val)
            val_loss = crit_reg(logits_val, yr_val).item()
            acc = (logits_val.argmax(dim=1) == yr_val).float().mean().item()
        print(f"[RegimeCNN] Epoch {epoch}/{REGIME_EPOCHS} "
              f"Train={train_loss:.6e} Val={val_loss:.6e} Acc={acc:.3f}")

    return lstm_model, tf_model, regime_model

# ================== フィーダー群 & 板情報フック ==================

def build_feeders(prices: np.ndarray,
                  lstm_model, tf_model, regime_model):
    returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)

    feat_mat = np.stack(
        [returns, vol_12, vol_36, trend_36, rsi, returns_smooth],
        axis=1
    )

    def _predict_forecaster(model, t):
        if t < LSTM_SEQ_LEN:
            return np.zeros(N_HORIZON, dtype=np.float32)
        end = t + 1
        start = end - LSTM_SEQ_LEN
        x = feat_mat[start:end]
        x_t = torch.tensor(x, dtype=torch.float32,
                           device=device).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred = model(x_t).squeeze(0).cpu().numpy()
        return pred.astype(np.float32)

    def lstm_feeder(t: int) -> np.ndarray:
        return _predict_forecaster(lstm_model, t)

    def tf_feeder(t: int) -> np.ndarray:
        return _predict_forecaster(tf_model, t)

    def regime_feeder(t: int) -> np.ndarray:
        if t < REGIME_SEQ_LEN:
            return np.ones(3, dtype=np.float32) / 3.0
        end = t
        start = end - REGIME_SEQ_LEN
        win = returns[start:end]
        w_t = torch.tensor(win, dtype=torch.float32,
                           device=device).unsqueeze(0)
        regime_model.eval()
        with torch.no_grad():
            logits = regime_model(w_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs.astype(np.float32)

    return lstm_feeder, tf_feeder, regime_feeder, returns, vol_12, vol_36, trend_36, rsi


def dummy_orderbook_feature(t: int) -> np.ndarray:
    """
    板データがないので今はゼロベクトルを返す。
    実際の板データを使うときはここを書き換えれば良い。
    例: [bid_volume, ask_volume, imbalance] など。
    """
    return np.zeros(3, dtype=np.float32)

# ================== 環境 ==================

class HybridEnv:
    def __init__(self, prices, feeders, orderbook_feature_func=None):
        (
            self.lstm_feeder,
            self.tf_feeder,
            self.regime_feeder,
            returns,
            vol_12,
            vol_36,
            trend_36,
            rsi,
        ) = feeders

        self.orderbook_feature_func = orderbook_feature_func or dummy_orderbook_feature

        self.prices = prices.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.vol_12 = vol_12.astype(np.float32)
        self.vol_36 = vol_36.astype(np.float32)
        self.trend_36 = trend_36.astype(np.float32)
        self.rsi = rsi.astype(np.float32)

        self.state_ret_len = STATE_RET_LEN
        self.reset_idx = STATE_RET_LEN
        self.max_t = len(self.returns) - 1
        self.position = 0
        self.t = None

        self.fusion_net = ForecastFusionNet().to(device)

        print("[HybridEnv] init: len(returns) =", len(self.returns))

    @property
    def state_dim(self):
        # ret_window + vol + trend + rsi + regime(3) + fused(N_HORIZON) + orderbook(3) + pos
        return self.state_ret_len + 1 + 1 + 1 + 3 + N_HORIZON + 3 + 1

    def reset(self):
        self.t = self.reset_idx
        self.position = 0
        return self._get_state()

    def _get_state(self):
        start = self.t - self.state_ret_len
        ret_window = self.returns[start:self.t]

        vol = self.vol_12[self.t]
        trend = self.trend_36[self.t]
        rsi_val = self.rsi[self.t]

        regime_probs = self.regime_feeder(self.t)  # (3,)
        lstm_pred = self.lstm_feeder(self.t)
        tf_pred   = self.tf_feeder(self.t)

        # 融合
        lstm_t = torch.tensor(lstm_pred, dtype=torch.float32,
                              device=device).unsqueeze(0)
        tf_t   = torch.tensor(tf_pred, dtype=torch.float32,
                              device=device).unsqueeze(0)
        reg_t  = torch.tensor(regime_probs, dtype=torch.float32,
                              device=device).unsqueeze(0)
        with torch.no_grad():
            fused = self.fusion_net(lstm_t, tf_t, reg_t).squeeze(0).cpu().numpy()

        # 板情報
        ob_feat = self.orderbook_feature_func(self.t)  # (3,)

        pos_scaled = self.position / MAX_POSITION

        state = np.concatenate(
            [
                ret_window,
                np.array([vol, trend, rsi_val], dtype=np.float32),
                regime_probs,
                fused,
                ob_feat,
                np.array([pos_scaled], dtype=np.float32),
            ]
        )
        return state.astype(np.float32)

    def step(self, action_idx):
        action_to_pos = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.int32)
        new_pos = int(action_to_pos[action_idx])

        prev_pos = self.position
        self.position = new_pos
        pos_change = abs(self.position - prev_pos)

        r = self.returns[self.t]
        price = self.prices[self.t]

        # 生PnL
        pnl = self.position * r

        # ボリュームコスト
        volume_cost = pos_change * TRANSACTION_COST

        # スプレッド & スリッページコスト（新規ポジ時のみ）
        spread_return   = (SPREAD_PIPS   * PIP_VALUE_JPY) / price
        slippage_return = (SLIPPAGE_PIPS * PIP_VALUE_JPY) / price
        if pos_change > 0:
            spread_slip_cost = abs(self.position) * (spread_return + slippage_return)
        else:
            spread_slip_cost = 0.0

        cost = volume_cost + spread_slip_cost
        reward = pnl - cost

        # 損失はやや強調
        if reward < 0:
            reward *= LOSS_FACTOR

        # トレンドブースト
        trend = self.trend_36[self.t]
        if abs(trend) > TREND_THRESHOLD:
            reward *= TREND_BOOST

        # 予測方向にポジると追加報酬
        lstm_pred = self.lstm_feeder(self.t)
        tf_pred   = self.tf_feeder(self.t)
        reward += LSTM_REWARD_SCALE * lstm_pred[0] * self.position
        reward += TF_REWARD_SCALE   * tf_pred[0]   * self.position

        self.t += 1
        done = self.t >= self.max_t
        next_state = self._get_state() if not done else None

        return next_state, float(reward), done, {}

# ================== PPO 関連 ==================

def collect_trajectory(env: HybridEnv, model: ActorCritic, steps_per_ep: int):
    model.eval()
    state = env.reset()

    states, actions, rewards, dones, logps, values = [], [], [], [], [], []

    for _ in range(steps_per_ep):
        s_t = torch.tensor(state, dtype=torch.float32,
                           device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(s_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)

        next_state, reward, done, _ = env.step(int(action.item()))

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        logps.append(logp.item())
        values.append(value.item())

        state = next_state
        if done:
            break

    if not dones[-1]:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32,
                               device=device).unsqueeze(0)
            _, last_value = model(s_t)
            last_value = last_value.item()
    else:
        last_value = 0.0

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.bool_),
        np.array(logps, dtype=np.float32),
        np.array(values, dtype=np.float32),
        float(last_value),
    )


def compute_gae(rewards, dones, values, last_value,
                gamma=GAMMA, lam=LAMBDA_GAE):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        next_val = last_value if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def train_with_config(envs: List[HybridEnv], config: Dict) -> Tuple[float, ActorCritic]:
    global LOSS_FACTOR, TREND_BOOST

    LOSS_FACTOR = config["loss_factor"]
    TREND_BOOST = config["trend_boost"]
    lr = config["lr"]

    state_dim = envs[0].state_dim
    model = ActorCritic(state_dim, N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_rewards = []
    total_episodes = EPISODES_PER_PAIR * len(envs)

    for ep in range(1, total_episodes + 1):
        env = np.random.choice(envs)
        (
            states, actions, rewards, dones,
            old_logp, values, last_val
        ) = collect_trajectory(env, model, STEPS_PER_EP)

        adv, ret = compute_gae(rewards, dones, values, last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        dataset_size = states_t.size(0)
        ep_reward = float(np.sum(rewards))
        episode_rewards.append(ep_reward)

        for _ in range(EPOCHS_PPO):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, MINI_BATCH):
                end = start + MINI_BATCH
                mb_idx = idx[start:end]

                mb_s = states_t[mb_idx]
                mb_a = actions_t[mb_idx]
                mb_old = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits, values_pred = model(mb_s)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(mb_a)

                ratio = torch.exp(logp - mb_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS,
                                    1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values_pred, mb_ret)
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"[PPO cfg={config}] Ep {ep}/{total_episodes} "
              f"reward={ep_reward:.4f}")

    avg_last = float(np.mean(episode_rewards[-len(envs):]))
    return avg_last, model

# ================== メタサーチ ==================

def meta_search(envs: List[HybridEnv], trials: int = 5) -> Tuple[Dict, ActorCritic]:
    """
    簡易ランダムサーチ:
      - lr: 10^-4.2 ~ 10^-3.2
      - loss_factor: 1.0 ~ 1.6
      - trend_boost: 1.2 ~ 3.0
    """
    best_cfg = None
    best_score = -1e9
    best_model = None

    print("\n=== Meta Search Start ===")
    for i in range(trials):
        cfg = {
            "lr": float(10 ** np.random.uniform(-4.2, -3.2)),
            "loss_factor": float(np.random.uniform(1.0, 1.6)),
            "trend_boost": float(np.random.uniform(1.2, 3.0)),
        }
        score, model = train_with_config(envs, cfg)
        print(f"[Meta] trial={i+1}/{trials} cfg={cfg} -> score={score:.4f}")
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_model = model

    print(f"=== Meta Search Done. Best cfg={best_cfg} score={best_score:.4f} ===")
    return best_cfg, best_model

# ================== シミュレーション（推論モード） ==================

def run_simulation(env: HybridEnv, model: ActorCritic,
                   steps: int = 500, log_interval: int = 50):
    model.eval()
    state = env.reset()
    equity = 1.0
    eq_curve = [equity]

    print("\n=== Simulation Start ===")
    for t in range(steps):
        s_t = torch.tensor(state, dtype=torch.float32,
                           device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(s_t)
            action = torch.argmax(logits, dim=-1).item()

        next_state, reward, done, _ = env.step(action)
        equity *= (1.0 + reward)
        eq_curve.append(equity)

        if t % log_interval == 0 or done:
            print(f"[Sim] t={t} action={action} pos={env.position} "
                  f"reward={reward:.6f} equity={equity:.4f}")

        state = next_state
        if done:
            break

    plt.figure(figsize=(10, 4))
    plt.plot(eq_curve)
    plt.title("Simulation Equity Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig("simulation_equity_curve.png")
    print("Saved: simulation_equity_curve.png")

# ================== メイン ==================

def main():
    # 1. 各ペアで forecaster & regime を学習し、環境を作る
    envs = []
    for csv in PAIR_CSV_LIST:
        df = load_close_series(csv)
        prices = df["close"].values
        print(f"\n=== Train forecasters/regime for {csv} ===")
        lstm_m, tf_m, reg_m = train_forecasters_and_regime(prices)
        feeders = build_feeders(prices, lstm_m, tf_m, reg_m)
        env = HybridEnv(prices, feeders, orderbook_feature_func=dummy_orderbook_feature)
        envs.append(env)

    # 2. メタサーチでベストハイパラ＆モデルを探す
    best_cfg, best_model = meta_search(envs, trials=5)

    # 3. 代表ペアでシミュレーション実行
    run_simulation(envs[0], best_model, steps=600, log_interval=60)


if __name__ == "__main__":
    print("=== Quant Meta Hybrid Trader v2 (ALL-IN) ===")
    main()
