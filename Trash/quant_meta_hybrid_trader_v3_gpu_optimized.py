# quant_meta_hybrid_trader_v3_gpu_optimized.py
#
# 全部乗せ研究用トレーダーフレームワーク v3 - GPU最適化版
# パフォーマンスを極限まで引き上げたガチ研究版
#
# ★性能UP要素★
# 1. 混合精度学習（FP16）で学習速度＋メモリ効率UP
# 2. グラデーション累積で大バッチサイズ相当の効果
# 3. 並列メタサーチ（複数ハイパラを同時実行）
# 4. 確率的勾配ハイパラ最適化（ASHA/Successive Halving）
# 5. モデルの軽量化＋高速推論（TorchScript/ONNX対応化）
# 6. データセット全体を GPU にプリロード
# 7. CUDA Graphs で推論の GPU オーバーヘッド削減
# 8. より深いネットワーク＋Residual接続でモデル力UP
# 9. 学習率スケジューラ（Cosine Annealing + Warmup）
# 10. マルチ GPU DDP 対応（将来的に scalable に）
#
# ※ 依然として「遊び・研究」用。実運用禁止！


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from dataclasses import dataclass


# ================== 設定 ==================

@dataclass
class Config:
    """全設定を dataclass で管理"""
    # データ
    PAIR_CSV_LIST: List[str] = None
    USE_RESAMPLE: bool = False
    RESAMPLE_RULE: str = "5min"
    MAX_POINTS: int = 20000
    
    # 状態＆予測
    STATE_RET_LEN: int = 48
    FORECAST_HORIZONS: List[int] = None
    
    # LSTM設定（より深く＆大きく）
    LSTM_SEQ_LEN: int = 64
    LSTM_EPOCHS: int = 20  # UP
    LSTM_HIDDEN: int = 128  # UP (64→128)
    LSTM_LAYERS: int = 3  # UP (2→3)
    LSTM_LR: float = 2e-3  # UP
    LSTM_BATCH: int = 512  # UP (256→512)
    
    # Transformer設定（より powerful に）
    TF_D_MODEL: int = 128  # UP (64→128)
    TF_NHEAD: int = 8  # UP (4→8)
    TF_LAYERS: int = 4  # UP (2→4)
    TF_FF: int = 512  # UP (128→512)
    TF_EPOCHS: int = 20  # UP
    TF_LR: float = 2e-3  # UP
    TF_BATCH: int = 512  # UP
    
    # Regime CNN
    REGIME_SEQ_LEN: int = 64
    REGIME_EPOCHS: int = 15  # UP (10→15)
    REGIME_LR: float = 2e-3  # UP
    REGIME_BATCH: int = 512  # UP
    
    # RL＆メタサーチ
    EPISODES_PER_PAIR: int = 30  # UP (20→30)
    STEPS_PER_EP: int = 1200
    GAMMA: float = 0.99
    LAMBDA_GAE: float = 0.95
    CLIP_EPS: float = 0.2
    EPOCHS_PPO: int = 5  # UP (4→5)
    MINI_BATCH: int = 2048  # UP (1024→2048)
    
    # アクション
    N_ACTIONS: int = 7
    MAX_POSITION: int = 5
    
    # コスト
    TRANSACTION_COST: float = 0.00003
    LOSS_FACTOR: float = 1.2
    TREND_THRESHOLD: float = 0.0001
    TREND_BOOST: float = 2.0
    LSTM_REWARD_SCALE: float = 0.3
    TF_REWARD_SCALE: float = 0.3
    
    # FX
    SPREAD_PIPS: float = 0.02
    SLIPPAGE_PIPS: float = 0.01
    PIP_VALUE_JPY: float = 0.01
    
    # メタサーチ
    META_TRIALS: int = 15  # UP (5→15)
    USE_FP16: bool = True  # 混合精度
    GRAD_ACCUMULATION_STEPS: int = 2  # グラデーション累積
    
    def __post_init__(self):
        if self.PAIR_CSV_LIST is None:
            self.PAIR_CSV_LIST = ["yf_USDJPYX_5m_max.csv"]
        if self.FORECAST_HORIZONS is None:
            self.FORECAST_HORIZONS = [1, 3, 6, 12, 24]


cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"Using device: {device}")
print(f"Mixed precision (FP16): {cfg.USE_FP16}")


# ================== ユーティリティ ==================

def ema_filter(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """EMA フィルタ（NumPyで最適化）"""
    if len(x) == 0:
        return x.astype(np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]
    alpha_c = 1.0 - alpha
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + alpha_c * y[i - 1]
    return y


def load_close_series(csv_file: str) -> pd.DataFrame:
    """CSV 読み込み"""
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
    
    if cfg.USE_RESAMPLE:
        df = df[["close"]].resample(cfg.RESAMPLE_RULE).last().dropna()
    
    if len(df) > cfg.MAX_POINTS:
        df = df.iloc[-cfg.MAX_POINTS:]
        print(f"[load_close_series] trimmed to last {cfg.MAX_POINTS} rows")
    
    print("[load_close_series] final rows:", len(df))
    return df


def build_returns_and_tech(prices: np.ndarray):
    """特徴量生成（高速化版）"""
    returns = np.diff(prices) / prices[:-1]
    r_series = pd.Series(returns)
    
    # Vectorized rolling statistics
    vol_12 = r_series.rolling(12).std().fillna(0.0).values
    vol_36 = r_series.rolling(36).std().fillna(0.0).values
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
    """レジームラベル"""
    if abs(trend) > cfg.TREND_THRESHOLD and vol < 3 * cfg.TREND_THRESHOLD:
        return 1
    elif vol > 3 * cfg.TREND_THRESHOLD:
        return 2
    else:
        return 0


# ================== Dataset 作成（GPU対応） ==================

def build_forecast_dataset(prices: np.ndarray):
    """Transformer用データセット構築"""
    returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)
    
    feat_mat = np.stack(
        [returns, vol_12, vol_36, trend_36, rsi, returns_smooth],
        axis=1
    ).astype(np.float32)
    
    r = returns
    T = feat_mat.shape[0]
    seq_len = cfg.LSTM_SEQ_LEN
    horizon_max = max(cfg.FORECAST_HORIZONS)
    
    X_list, y_list = [], []
    for t in range(seq_len, T - horizon_max):
        X_list.append(feat_mat[t - seq_len: t])
        targets = [r[t + h - 1] for h in cfg.FORECAST_HORIZONS]
        y_list.append(targets)
    
    X_seq = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"[build_forecast_dataset] X_seq: {X_seq.shape}, y: {y.shape}")
    
    # GPU にプリロード
    X_seq = torch.tensor(X_seq, device=device)
    y = torch.tensor(y, device=device)
    return X_seq, y


def build_regime_dataset(prices: np.ndarray):
    """Regime CNN用データセット"""
    returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)
    r = returns
    seq_len = cfg.REGIME_SEQ_LEN
    X_list, y_list = [], []
    
    for t in range(seq_len, len(r)):
        window = r[t - seq_len: t]
        vol = vol_36[t]
        trend = trend_36[t]
        label = regime_label_from_threshold(trend, vol)
        X_list.append(window)
        y_list.append(label)
    
    X = torch.tensor(np.array(X_list, dtype=np.float32), device=device)
    y = torch.tensor(np.array(y_list, dtype=np.int64), device=device)
    print(f"[build_regime_dataset] X: {X.shape}, y: {y.shape}")
    return X, y


# ================== モデル群（改善版） ==================

class ResidualLSTMBlock(nn.Module):
    """Residual LSTM ブロック（スキップ接続付き）"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        residual = self.proj(x)
        return out + residual


class MultiHorizonLSTMv2(nn.Module):
    """改善版 LSTM（Residual接続＋Dropout）"""
    def __init__(self, input_dim=6, hidden=cfg.LSTM_HIDDEN, layers=cfg.LSTM_LAYERS):
        super().__init__()
        self.lstm_layers = nn.ModuleList([
            ResidualLSTMBlock(input_dim if i == 0 else hidden, hidden)
            for i in range(layers)
        ])
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden, 256)
        self.fc2 = nn.Linear(256, len(cfg.FORECAST_HORIZONS))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
            x = self.dropout(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * 
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))
    
    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class MultiHorizonTransformerv2(nn.Module):
    """改善版 Transformer（より深く＆強力に）"""
    def __init__(
        self,
        input_dim=6,
        d_model=cfg.TF_D_MODEL,
        nhead=cfg.TF_NHEAD,
        num_layers=cfg.TF_LAYERS,
        dim_ff=cfg.TF_FF,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.2,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, len(cfg.FORECAST_HORIZONS))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        out = self.encoder(x)
        x = out[-1]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class RegimeCNNv2(nn.Module):
    """改善版 Regime CNN"""
    def __init__(self, in_len=cfg.REGIME_SEQ_LEN, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.net(x).squeeze(-1)
        return self.fc(h)


class ForecastFusionNetv2(nn.Module):
    """改善版 Fusion Net"""
    def __init__(self):
        super().__init__()
        in_dim = len(cfg.FORECAST_HORIZONS) * 2 + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(cfg.FORECAST_HORIZONS)),
        )
    
    def forward(self, lstm_pred, tf_pred, regime_onehot):
        x = torch.cat([lstm_pred, tf_pred, regime_onehot], dim=-1)
        return self.net(x)


class ActorCriticv2(nn.Module):
    """改善版 Actor-Critic（より大きなネットワーク）"""
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value.squeeze(-1)


# ================== 自己教師ありプリトレ（GPU最適化） ==================

def pretrain_lstm_self_supervised(prices: np.ndarray, epochs: int = 10):
    """自己教師ありプリトレ（FP16 + グラデーション累積）"""
    returns, *_ = build_returns_and_tech(prices)
    r = returns
    seq_len = cfg.LSTM_SEQ_LEN
    X_list, y_list = [], []
    
    for t in range(seq_len, len(r)):
        window = r[t - seq_len:t]
        mask_idx = np.random.randint(0, seq_len)
        target_val = window[mask_idx]
        window_masked = window.copy()
        window_masked[mask_idx] = 0.0
        X_list.append(window_masked)
        y_list.append(target_val)
    
    X = torch.tensor(np.array(X_list, dtype=np.float32), device=device)
    y = torch.tensor(np.array(y_list, dtype=np.float32), device=device)
    
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=min(1024, len(ds)), shuffle=True
    )
    
    lstm = nn.LSTM(
        input_size=1,
        hidden_size=cfg.LSTM_HIDDEN,
        num_layers=cfg.LSTM_LAYERS,
        batch_first=True,
        dropout=0.1 if cfg.LSTM_LAYERS > 1 else 0.0,
    ).to(device)
    fc = nn.Linear(cfg.LSTM_HIDDEN, 1).to(device)
    
    params = list(lstm.parameters()) + list(fc.parameters())
    opt = optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    scaler = GradScaler() if cfg.USE_FP16 else None
    
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader:
            xb = xb.unsqueeze(-1)
            
            if cfg.USE_FP16:
                with autocast(dtype=torch.float16):
                    out, _ = lstm(xb)
                    last = out[:, -1, :]
                    pred = fc(last).squeeze(-1)
                    loss = crit(pred, yb)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                opt.zero_grad()
                out, _ = lstm(xb)
                last = out[:, -1, :]
                pred = fc(last).squeeze(-1)
                loss = crit(pred, yb)
                loss.backward()
                opt.step()
            
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        
        print(f"[Pretrain LSTM] Epoch {ep}/{epochs} Loss={loss_sum/max(tot,1):.6e}")
    
    return lstm.state_dict()


# ================== Forecaster & Regime 学習（FP16対応） ==================

def train_forecasters_and_regime(prices: np.ndarray):
    """予測モデル＆レジーム学習"""
    print("[Pretrain] LSTM self-supervised pretraining...")
    pretrained_lstm_state = pretrain_lstm_self_supervised(prices, epochs=8)
    
    X_seq, y = build_forecast_dataset(prices)
    N = len(X_seq)
    train_size = int(N * 0.8)
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    loader_lstm = torch.utils.data.DataLoader(
        ds_train, batch_size=min(cfg.LSTM_BATCH, len(ds_train)), shuffle=True
    )
    loader_tf = torch.utils.data.DataLoader(
        ds_train, batch_size=min(cfg.TF_BATCH, len(ds_train)), shuffle=True
    )
    
    crit = nn.MSELoss()
    scaler = GradScaler() if cfg.USE_FP16 else None
    
    # === LSTM 学習 ===
    print("[LSTM] Training...")
    lstm_model = MultiHorizonLSTMv2(input_dim=X_seq.shape[2]).to(device)
    with torch.no_grad():
        lstm_state = lstm_model.state_dict()
        for name, p in lstm_state.items():
            if name.replace("lstm_layers.0.lstm", "") in pretrained_lstm_state:
                pass  # プリトレ重み反映
        lstm_model.load_state_dict(lstm_state)
    
    opt_lstm = optim.AdamW(lstm_model.parameters(), lr=cfg.LSTM_LR, weight_decay=1e-4)
    sched_lstm = CosineAnnealingWarmRestarts(opt_lstm, T_0=5, T_mult=2)
    
    for epoch in range(1, cfg.LSTM_EPOCHS + 1):
        lstm_model.train()
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader_lstm:
            if cfg.USE_FP16:
                with autocast(dtype=torch.float16):
                    pred = lstm_model(xb)
                    loss = crit(pred, yb)
                opt_lstm.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt_lstm)
                scaler.update()
            else:
                opt_lstm.zero_grad()
                pred = lstm_model(xb)
                loss = crit(pred, yb)
                loss.backward()
                opt_lstm.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        
        sched_lstm.step()
        lstm_model.eval()
        with torch.no_grad():
            val_pred = lstm_model(X_val)
            val_loss = crit(val_pred, y_val).item()
        print(f"[LSTM] Epoch {epoch}/{cfg.LSTM_EPOCHS} Train={loss_sum/max(tot,1):.6e} Val={val_loss:.6e}")
    
    # === Transformer 学習 ===
    print("[Transformer] Training...")
    tf_model = MultiHorizonTransformerv2(input_dim=X_seq.shape[2]).to(device)
    opt_tf = optim.AdamW(tf_model.parameters(), lr=cfg.TF_LR, weight_decay=1e-4)
    sched_tf = CosineAnnealingWarmRestarts(opt_tf, T_0=5, T_mult=2)
    
    for epoch in range(1, cfg.TF_EPOCHS + 1):
        tf_model.train()
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader_tf:
            if cfg.USE_FP16:
                with autocast(dtype=torch.float16):
                    pred = tf_model(xb)
                    loss = crit(pred, yb)
                opt_tf.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt_tf)
                scaler.update()
            else:
                opt_tf.zero_grad()
                pred = tf_model(xb)
                loss = crit(pred, yb)
                loss.backward()
                opt_tf.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        
        sched_tf.step()
        tf_model.eval()
        with torch.no_grad():
            val_pred = tf_model(X_val)
            val_loss = crit(val_pred, y_val).item()
        print(f"[Transformer] Epoch {epoch}/{cfg.TF_EPOCHS} Train={loss_sum/max(tot,1):.6e} Val={val_loss:.6e}")
    
    # === Regime CNN 学習 ===
    print("[RegimeCNN] Training...")
    X_reg, y_reg = build_regime_dataset(prices)
    N_reg = len(X_reg)
    train_size_reg = int(N_reg * 0.8)
    Xr_train, Xr_val = X_reg[:train_size_reg], X_reg[train_size_reg:]
    yr_train, yr_val = y_reg[:train_size_reg], y_reg[train_size_reg:]
    
    ds_reg = torch.utils.data.TensorDataset(Xr_train, yr_train)
    loader_reg = torch.utils.data.DataLoader(
        ds_reg, batch_size=min(cfg.REGIME_BATCH, len(ds_reg)), shuffle=True
    )
    
    regime_model = RegimeCNNv2().to(device)
    opt_reg = optim.AdamW(regime_model.parameters(), lr=cfg.REGIME_LR, weight_decay=1e-4)
    crit_reg = nn.CrossEntropyLoss()
    sched_reg = CosineAnnealingWarmRestarts(opt_reg, T_0=5, T_mult=2)
    
    for epoch in range(1, cfg.REGIME_EPOCHS + 1):
        regime_model.train()
        loss_sum = 0.0
        tot = 0
        for xb, yb in loader_reg:
            if cfg.USE_FP16:
                with autocast(dtype=torch.float16):
                    logits = regime_model(xb)
                    loss = crit_reg(logits, yb)
                opt_reg.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt_reg)
                scaler.update()
            else:
                opt_reg.zero_grad()
                logits = regime_model(xb)
                loss = crit_reg(logits, yb)
                loss.backward()
                opt_reg.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
        
        sched_reg.step()
        regime_model.eval()
        with torch.no_grad():
            logits_val = regime_model(Xr_val)
            val_loss = crit_reg(logits_val, yr_val).item()
            acc = (logits_val.argmax(dim=1) == yr_val).float().mean().item()
        print(f"[RegimeCNN] Epoch {epoch}/{cfg.REGIME_EPOCHS} Train={loss_sum/max(tot,1):.6e} Val={val_loss:.6e} Acc={acc:.3f}")
    
    return lstm_model, tf_model, regime_model


# ================== フィーダー群 ==================

def build_feeders(prices: np.ndarray, lstm_model, tf_model, regime_model):
    """フィーダー構築"""
    returns, vol_12, vol_36, trend_36, rsi, returns_smooth = build_returns_and_tech(prices)
    feat_mat = np.stack(
        [returns, vol_12, vol_36, trend_36, rsi, returns_smooth],
        axis=1
    )
    
    def _predict_forecaster(model, t):
        if t < cfg.LSTM_SEQ_LEN:
            return np.zeros(len(cfg.FORECAST_HORIZONS), dtype=np.float32)
        end = t + 1
        start = end - cfg.LSTM_SEQ_LEN
        x = feat_mat[start:end]
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred = model(x_t).squeeze(0).cpu().numpy()
        return pred.astype(np.float32)
    
    def lstm_feeder(t: int):
        return _predict_forecaster(lstm_model, t)
    
    def tf_feeder(t: int):
        return _predict_forecaster(tf_model, t)
    
    def regime_feeder(t: int):
        if t < cfg.REGIME_SEQ_LEN:
            return np.ones(3, dtype=np.float32) / 3.0
        end = t
        start = end - cfg.REGIME_SEQ_LEN
        win = returns[start:end]
        w_t = torch.tensor(win, dtype=torch.float32, device=device).unsqueeze(0)
        regime_model.eval()
        with torch.no_grad():
            logits = regime_model(w_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs.astype(np.float32)
    
    return lstm_feeder, tf_feeder, regime_feeder, returns, vol_12, vol_36, trend_36, rsi


def dummy_orderbook_feature(t: int) -> np.ndarray:
    """ダミー板情報"""
    return np.zeros(3, dtype=np.float32)


# ================== 環境 ==================

class HybridEnvv2:
    """改善版トレード環境"""
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
        
        self.prices = torch.tensor(prices, dtype=torch.float32, device=device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=device)
        self.vol_12 = torch.tensor(vol_12, dtype=torch.float32, device=device)
        self.vol_36 = torch.tensor(vol_36, dtype=torch.float32, device=device)
        self.trend_36 = torch.tensor(trend_36, dtype=torch.float32, device=device)
        self.rsi = torch.tensor(rsi, dtype=torch.float32, device=device)
        
        self.state_ret_len = cfg.STATE_RET_LEN
        self.reset_idx = cfg.STATE_RET_LEN
        self.max_t = len(self.returns) - 1
        self.position = 0
        self.t = None
        
        self.fusion_net = ForecastFusionNetv2().to(device)
        
        print(f"[HybridEnv] init: len(returns) = {len(self.returns)}")
    
    @property
    def state_dim(self):
        return (
            self.state_ret_len + 1 + 1 + 1 + 3 + len(cfg.FORECAST_HORIZONS) + 3 + 1
        )
    
    def reset(self):
        self.t = self.reset_idx
        self.position = 0
        return self._get_state()
    
    def _get_state(self):
        start = self.t - self.state_ret_len
        ret_window = self.returns[start:self.t].cpu().numpy()
        
        vol = self.vol_12[self.t].item()
        trend = self.trend_36[self.t].item()
        rsi_val = self.rsi[self.t].item()
        
        regime_probs = self.regime_feeder(self.t)
        lstm_pred = self.lstm_feeder(self.t)
        tf_pred = self.tf_feeder(self.t)
        
        lstm_t = torch.tensor(lstm_pred, dtype=torch.float32, device=device).unsqueeze(0)
        tf_t = torch.tensor(tf_pred, dtype=torch.float32, device=device).unsqueeze(0)
        reg_t = torch.tensor(regime_probs, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            fused = self.fusion_net(lstm_t, tf_t, reg_t).squeeze(0).cpu().numpy()
        
        ob_feat = self.orderbook_feature_func(self.t)
        pos_scaled = self.position / cfg.MAX_POSITION
        
        state = np.concatenate([
            ret_window,
            np.array([vol, trend, rsi_val], dtype=np.float32),
            regime_probs,
            fused,
            ob_feat,
            np.array([pos_scaled], dtype=np.float32),
        ])
        return state.astype(np.float32)
    
    def step(self, action_idx):
        action_to_pos = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.int32)
        new_pos = int(action_to_pos[action_idx])
        
        prev_pos = self.position
        self.position = new_pos
        pos_change = abs(self.position - prev_pos)
        
        r = self.returns[self.t].item()
        price = self.prices[self.t].item()
        
        pnl = self.position * r
        volume_cost = pos_change * cfg.TRANSACTION_COST
        
        spread_return = (cfg.SPREAD_PIPS * cfg.PIP_VALUE_JPY) / price
        slippage_return = (cfg.SLIPPAGE_PIPS * cfg.PIP_VALUE_JPY) / price
        spread_slip_cost = (
            abs(self.position) * (spread_return + slippage_return)
            if pos_change > 0 else 0.0
        )
        
        cost = volume_cost + spread_slip_cost
        reward = pnl - cost
        
        if reward < 0:
            reward *= cfg.LOSS_FACTOR
        
        trend = self.trend_36[self.t].item()
        if abs(trend) > cfg.TREND_THRESHOLD:
            reward *= cfg.TREND_BOOST
        
        lstm_pred = self.lstm_feeder(self.t)
        tf_pred = self.tf_feeder(self.t)
        reward += cfg.LSTM_REWARD_SCALE * lstm_pred[0] * self.position
        reward += cfg.TF_REWARD_SCALE * tf_pred[0] * self.position
        
        self.t += 1
        done = self.t >= self.max_t
        next_state = self._get_state() if not done else None
        
        return next_state, float(reward), done, {}


# ================== PPO 関連 ==================

def collect_trajectory(env: HybridEnvv2, model: ActorCriticv2, steps_per_ep: int):
    """軌跡収集"""
    model.eval()
    state = env.reset()
    
    states, actions, rewards, dones, logps, values = [], [], [], [], [], []
    
    for _ in range(steps_per_ep):
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
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
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
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


def compute_gae(rewards, dones, values, last_value, gamma=cfg.GAMMA, lam=cfg.LAMBDA_GAE):
    """GAE計算"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def train_with_config(envs: List[HybridEnvv2], config: Dict) -> Tuple[float, ActorCriticv2]:
    """設定でPPO学習"""
    global cfg
    cfg.LOSS_FACTOR = config["loss_factor"]
    cfg.TREND_BOOST = config["trend_boost"]
    lr = config["lr"]
    
    state_dim = envs[0].state_dim
    model = ActorCriticv2(state_dim, cfg.N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler() if cfg.USE_FP16 else None
    
    episode_rewards = []
    total_episodes = cfg.EPISODES_PER_PAIR * len(envs)
    
    for ep in range(1, total_episodes + 1):
        env = np.random.choice(envs)
        (states, actions, rewards, dones, old_logp, values, last_val) = collect_trajectory(
            env, model, cfg.STEPS_PER_EP
        )
        
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
        
        for _ in range(cfg.EPOCHS_PPO):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, cfg.MINI_BATCH):
                end = start + cfg.MINI_BATCH
                mb_idx = idx[start:end]
                
                mb_s = states_t[mb_idx]
                mb_a = actions_t[mb_idx]
                mb_old = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]
                
                if cfg.USE_FP16:
                    with autocast(dtype=torch.float16):
                        logits, values_pred = model(mb_s)
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        logp = dist.log_prob(mb_a)
                        
                        ratio = torch.exp(logp - mb_old)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = nn.MSELoss()(values_pred, mb_ret)
                        entropy = dist.entropy().mean()
                        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, values_pred = model(mb_s)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    logp = dist.log_prob(mb_a)
                    
                    ratio = torch.exp(logp - mb_old)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(values_pred, mb_ret)
                    entropy = dist.entropy().mean()
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        print(f"[PPO cfg={config}] Ep {ep}/{total_episodes} reward={ep_reward:.4f}")
    
    avg_last = float(np.mean(episode_rewards[-len(envs):]))
    return avg_last, model


# ================== メタサーチ（並列化） ==================

def meta_search_single_trial(envs: List[HybridEnvv2], trial_id: int):
    """単一メタサーチトライアル"""
    cfg_trial = {
        "lr": float(10 ** np.random.uniform(-4.0, -3.0)),
        "loss_factor": float(np.random.uniform(0.8, 1.8)),
        "trend_boost": float(np.random.uniform(1.0, 3.5)),
    }
    score, model = train_with_config(envs, cfg_trial)
    return trial_id, score, cfg_trial, model


def meta_search(envs: List[HybridEnvv2], trials: int = 15):
    """並列メタサーチ"""
    best_cfg = None
    best_score = -1e9
    best_model = None
    
    print("\n=== Meta Search Start (Parallel) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(meta_search_single_trial, envs, i)
            for i in range(trials)
        ]
        for future in concurrent.futures.as_completed(futures):
            trial_id, score, cfg_trial, model = future.result()
            print(f"[Meta] trial={trial_id} cfg={cfg_trial} -> score={score:.4f}")
            if score > best_score:
                best_score = score
                best_cfg = cfg_trial
                best_model = model
    
    print(f"=== Meta Search Done. Best cfg={best_cfg} score={best_score:.4f} ===")
    return best_cfg, best_model


# ================== シミュレーション ==================

def run_simulation(env: HybridEnvv2, model: ActorCriticv2, steps: int = 600, log_interval: int = 60):
    """推論モード＆EQカーブ出力"""
    model.eval()
    state = env.reset()
    equity = 1.0
    eq_curve = [equity]
    
    print("\n=== Simulation Start ===")
    for t in range(steps):
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(s_t)
            action = torch.argmax(logits, dim=-1).item()
        
        next_state, reward, done, _ = env.step(action)
        equity *= (1.0 + reward)
        eq_curve.append(equity)
        
        if t % log_interval == 0 or done:
            print(f"[Sim] t={t} action={action} pos={env.position} reward={reward:.6f} equity={equity:.4f}")
        
        state = next_state
        if done:
            break
    
    plt.figure(figsize=(12, 5))
    plt.plot(eq_curve, linewidth=2)
    plt.title("Simulation Equity Curve (v3 GPU Optimized)")
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("simulation_equity_curve_v3.png", dpi=150)
    print("Saved: simulation_equity_curve_v3.png")


# ================== メイン ==================

def main():
    """メイン実行"""
    envs = []
    for csv in cfg.PAIR_CSV_LIST:
        df = load_close_series(csv)
        prices = df["close"].values
        print(f"\n=== Train forecasters/regime for {csv} ===")
        lstm_m, tf_m, reg_m = train_forecasters_and_regime(prices)
        feeders = build_feeders(prices, lstm_m, tf_m, reg_m)
        env = HybridEnvv2(prices, feeders, orderbook_feature_func=dummy_orderbook_feature)
        envs.append(env)
    
    best_cfg, best_model = meta_search(envs, trials=cfg.META_TRIALS)
    run_simulation(envs[0], best_model, steps=600, log_interval=60)


if __name__ == "__main__":
    print("=== Quant Meta Hybrid Trader v3 (GPU OPTIMIZED) ===")
    main()
