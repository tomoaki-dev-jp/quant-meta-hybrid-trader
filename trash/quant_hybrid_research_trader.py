# quant_hybrid_research_trader.py
#
# noriちん用・金融工学 / RL / 時系列研究フレームワーク
#
# ・複数通貨ペアのCSVを読み込み（yfinance形式を想定）
# ・LSTMで +1, +3, +6 ステップ先リターンを予測
# ・Transformerでも +1, +3, +6 を予測（LSTMと別系統）
# ・レジーム判定（trend/vol から 3クラス: Trend, Range, HighVol）
# ・LSTM予測 + Transformer予測 + Regime + テクニカル を
#   PPOエージェントの状態として入力
# ・通貨ペアはエピソードごとにランダムに選択（マルチアセット学習）
#
# ※研究・遊び用途限定。実運用禁止！

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple

# ================== 設定 ==================

# ここに複数通貨ペアのCSVパスを並べる
PAIR_CSV_LIST = [
    "fx/yf_USDJPYX_5m_max.csv",
    # "fx/yf_EURUSD_5m_max.csv",
    # "fx/yf_GBPUSD_5m_max.csv",
]

USE_RESAMPLE  = False
RESAMPLE_RULE = "5min"
MAX_POINTS    = 20000

# ---- 時系列 / テクニカル ----
STATE_RET_LEN = 48

# ---- LSTM / Transformer 予測 ----
FORECAST_HORIZONS = [1, 3, 6]   # +1, +3, +6 ステップ先
N_HORIZON = len(FORECAST_HORIZONS)

LSTM_SEQ_LEN  = 64
LSTM_EPOCHS   = 20
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
LSTM_LR       = 1e-3
LSTM_BATCH    = 256

TF_D_MODEL    = 64
TF_NHEAD      = 4
TF_LAYERS     = 2
TF_FF         = 128
TF_EPOCHS     = 20
TF_LR         = 1e-3
TF_BATCH      = 256

# ---- PPO / RL ----
EPISODES_PER_PAIR = 40    # 各ペアあたりのエピソード数
STEPS_PER_EP      = 1500

GAMMA        = 0.99
LAMBDA_GAE   = 0.95
LR_PPO       = 3e-4
CLIP_EPS     = 0.2
EPOCHS_PPO   = 4
MINI_BATCH   = 1024

N_ACTIONS    = 7       # [-5,-3,-1,0,1,3,5]
MAX_POSITION = 5

TRANSACTION_COST   = 0.00003
LOSS_FACTOR        = 1.2
TREND_THRESHOLD    = 0.0001
TREND_BOOST        = 2.0
LSTM_REWARD_SCALE  = 0.3
TF_REWARD_SCALE    = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print("Using device:", device)


# ================== 共通：データ読み込み ==================

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


# ================== テクニカル / レジーム判定 ==================

def build_returns_and_tech(prices: np.ndarray):
    returns = np.diff(prices) / prices[:-1]
    r_series = pd.Series(returns)

    vol_12   = r_series.rolling(12).std().fillna(0.0).values
    vol_36   = r_series.rolling(36).std().fillna(0.0).values
    trend_36 = r_series.rolling(36).mean().fillna(0.0).values

    up = r_series.clip(lower=0).rolling(14).mean()
    down = (-r_series.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 * up / (up + down + 1e-9)
    rsi = ((rsi - 50.0) / 50.0).fillna(0.0).values   # -1〜+1程度

    return returns.astype(np.float32), vol_12.astype(np.float32), \
        vol_36.astype(np.float32), trend_36.astype(np.float32), \
        rsi.astype(np.float32)


def regime_classifier(trend: float, vol: float) -> int:
    """
    ざっくり3クラスに分けるレジーム判定。
    0: Range (トレンド弱 & 低〜中ボラ)
    1: Trend (トレンド強)
    2: HighVol (高ボラ)
    """
    if abs(trend) > TREND_THRESHOLD and vol < 3 * TREND_THRESHOLD:
        return 1  # Trend
    elif vol > 3 * TREND_THRESHOLD:
        return 2  # HighVol
    else:
        return 0  # Range


# ================== LSTM / Transformer 用 Dataset ==================

def build_forecast_dataset(
    prices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LSTM/Transformer 共用の学習用データセットを作る。
    戻り値:
      X_seq: (N, seq_len, feat_dim)
      y_lstm: (N, N_HORIZON)  3ホライズンのターゲット
      mask_index: returns の index とXの対応（後でfeed用に使う）
    """
    returns, vol_12, vol_36, trend_36, rsi = build_returns_and_tech(prices)
    r = returns
    r_series = pd.Series(r)

    feat_mat = np.stack(
        [
            r,
            vol_12,
            vol_36,
            trend_36,
            rsi,
        ],
        axis=1,
    )  # (T-1, 5)

    T = feat_mat.shape[0]
    seq_len = LSTM_SEQ_LEN
    horizon_max = max(FORECAST_HORIZONS)

    X_list = []
    y_list = []
    idx_list = []

    for t in range(seq_len, T - horizon_max):
        X_list.append(feat_mat[t - seq_len : t])
        # 各ホライズンのターゲットリターン
        ys = []
        for h in FORECAST_HORIZONS:
            ys.append(r[t + h - 1])
        y_list.append(ys)
        idx_list.append(t)  # returns index t-1 付近

    X_seq = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    idx_arr = np.array(idx_list, dtype=np.int64)
    print("[build_forecast_dataset] X_seq:", X_seq.shape, "y:", y.shape)
    return X_seq, y, idx_arr


# ================== LSTM モデル ==================

class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS):
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
        return self.fc(last)  # (B, N_HORIZON)


# ================== Transformer モデル ==================

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
        self.pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len].to(x.device)


class MultiHorizonTransformer(nn.Module):
    def __init__(
        self,
        input_dim=5,
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
        # x: (B, seq_len, input_dim)
        x = self.input_proj(x)   # (B, seq_len, d_model)
        x = x.permute(1, 0, 2)   # (seq_len, B, d_model)
        x = self.pos_enc(x)
        out = self.encoder(x)
        last = out[-1]           # (B, d_model)
        return self.fc(last)     # (B, N_HORIZON)


# ================== 予測モデルの学習 ==================

def train_forecaster(
    prices: np.ndarray,
):
    X_seq, y, idx_arr = build_forecast_dataset(prices)
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

    # ---- LSTM ----
    lstm_model = MultiHorizonLSTM(input_dim=X_seq.shape[2]).to(device)
    opt_lstm = optim.AdamW(lstm_model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    crit = nn.MSELoss()

    for epoch in range(1, LSTM_EPOCHS + 1):
        lstm_model.train()
        total = 0
        loss_sum = 0.0
        for xb, yb in loader_lstm:
            opt_lstm.zero_grad()
            pred = lstm_model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt_lstm.step()
            loss_sum += loss.item() * len(xb)
            total += len(xb)
        train_loss = loss_sum / max(total, 1)

        lstm_model.eval()
        with torch.no_grad():
            val_pred = lstm_model(X_val)
            val_loss = crit(val_pred, y_val).item()
        print(f"[LSTM] Epoch {epoch}/{LSTM_EPOCHS} "
              f"Train={train_loss:.6e} Val={val_loss:.6e}")

    # ---- Transformer ----
    tf_model = MultiHorizonTransformer(input_dim=X_seq.shape[2]).to(device)
    opt_tf = optim.AdamW(tf_model.parameters(), lr=TF_LR, weight_decay=1e-4)

    for epoch in range(1, TF_EPOCHS + 1):
        tf_model.train()
        total = 0
        loss_sum = 0.0
        for xb, yb in loader_tf:
            opt_tf.zero_grad()
            pred = tf_model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt_tf.step()
            loss_sum += loss.item() * len(xb)
            total += len(xb)
        train_loss = loss_sum / max(total, 1)

        tf_model.eval()
        with torch.no_grad():
            val_pred = tf_model(X_val)
            val_loss = crit(val_pred, y_val).item()
        print(f"[Transformer] Epoch {epoch}/{TF_EPOCHS} "
              f"Train={train_loss:.6e} Val={val_loss:.6e}")

    return lstm_model, tf_model


def build_forecast_feeders(
    prices: np.ndarray, lstm_model: nn.Module, tf_model: nn.Module
):
    """
    returns index t に対して
      lstm_pred(t): shape (N_HORIZON,)
      tf_pred(t):   shape (N_HORIZON,)
    を返す関数を構築。
    """
    returns, vol_12, vol_36, trend_36, rsi = build_returns_and_tech(prices)

    feat_mat = np.stack(
        [returns, vol_12, vol_36, trend_36, rsi],
        axis=1,
    )  # (T-1, 5)

    def _predict_core(model, t):
        if t < LSTM_SEQ_LEN:
            return np.zeros(N_HORIZON, dtype=np.float32)
        end = t + 1
        start = end - LSTM_SEQ_LEN
        x = feat_mat[start:end]  # (LSTM_SEQ_LEN, 5)
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred = model(x_t).squeeze(0).cpu().numpy()
        return pred.astype(np.float32)

    def lstm_feeder(t_index: int) -> np.ndarray:
        return _predict_core(lstm_model, t_index)

    def tf_feeder(t_index: int) -> np.ndarray:
        return _predict_core(tf_model, t_index)

    return lstm_feeder, tf_feeder, returns, vol_12, vol_36, trend_36, rsi


# ================== HybridTradingEnv（マルチモデル＋レジーム） ==================

class HybridTradingEnv:
    """
    1つの通貨ペアの上でトレードする環境。
    状態:
      - 過去 STATE_RET_LEN 本のリターン
      - vol_12, trend_36, rsi (1step分)
      - regime (one-hot 3次元)
      - LSTM予測 (N_HORIZON)
      - Transformer予測 (N_HORIZON)
      - 現ポジション (スケーリング1要素)
    行動:
      0..6 -> position in [-5,-3,-1,0,1,3,5]
    """

    def __init__(
        self,
        prices: np.ndarray,
        lstm_feeder,
        tf_feeder,
        returns,
        vol_12,
        vol_36,
        trend_36,
        rsi,
    ):
        self.prices = prices.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.vol_12 = vol_12.astype(np.float32)
        self.vol_36 = vol_36.astype(np.float32)
        self.trend_36 = trend_36.astype(np.float32)
        self.rsi = rsi.astype(np.float32)

        self.lstm_feeder = lstm_feeder
        self.tf_feeder = tf_feeder

        self.state_ret_len = STATE_RET_LEN
        self.reset_idx = STATE_RET_LEN
        self.max_t = len(self.returns) - 1
        self.position = 0
        self.t = None

        print("[HybridTradingEnv] init:",
              f"len(prices)={len(self.prices)}, len(returns)={len(self.returns)}")

    @property
    def state_dim(self):
        # ret_window + vol + trend + rsi + regime(3) +
        # LSTM(N_HORIZON) + TF(N_HORIZON) + pos
        return (
            self.state_ret_len
            + 1
            + 1
            + 1
            + 3
            + N_HORIZON
            + N_HORIZON
            + 1
        )

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

        # regime one-hot
        reg = regime_classifier(trend, self.vol_36[self.t])
        regime_vec = np.zeros(3, dtype=np.float32)
        regime_vec[reg] = 1.0

        lstm_pred = self.lstm_feeder(self.t)      # (N_HORIZON,)
        tf_pred   = self.tf_feeder(self.t)        # (N_HORIZON,)

        pos_scaled = self.position / MAX_POSITION

        state = np.concatenate(
            [
                ret_window,
                np.array([vol, trend, rsi_val], dtype=np.float32),
                regime_vec,
                lstm_pred,
                tf_pred,
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
        pnl = self.position * r
        cost = pos_change * TRANSACTION_COST
        reward = pnl - cost

        if reward < 0:
            reward *= LOSS_FACTOR

        trend = self.trend_36[self.t]
        if abs(trend) > TREND_THRESHOLD:
            reward *= TREND_BOOST

        lstm_pred = self.lstm_feeder(self.t)
        tf_pred   = self.tf_feeder(self.t)

        # 一番近い(+1ステップ)予測を重みに使う（ベクトルでもいいけど簡略化）
        reward += LSTM_REWARD_SCALE * lstm_pred[0] * self.position
        reward += TF_REWARD_SCALE   * tf_pred[0]   * self.position

        self.t += 1
        done = self.t >= self.max_t
        next_state = self._get_state() if not done else None

        return next_state, float(reward), done, {}


# ================== PPO エージェント ==================

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


def collect_trajectory(env: HybridTradingEnv, model, steps_per_ep):
    model.eval()
    state = env.reset()

    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

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
        log_probs.append(logp.item())
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
        np.array(log_probs, dtype=np.float32),
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


def train_hybrid_ppo_multi_asset():
    # 各ペアごとに forecaster & env を用意
    pair_envs = []
    for csv in PAIR_CSV_LIST:
        df = load_close_series(csv)
        prices = df["close"].values
        print(f"\n=== Training forecasters for {csv} ===")
        lstm_model, tf_model = train_forecaster(prices)
        lstm_feeder, tf_feeder, returns, vol_12, vol_36, trend_36, rsi = \
            build_forecast_feeders(prices, lstm_model, tf_model)
        env = HybridTradingEnv(
            prices,
            lstm_feeder,
            tf_feeder,
            returns,
            vol_12,
            vol_36,
            trend_36,
            rsi,
        )
        pair_envs.append(env)

    # 状態次元はどのenvも同じ想定
    state_dim = pair_envs[0].state_dim
    model = ActorCritic(state_dim, N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_PPO)

    episode_rewards = []
    last_equity_curve = None

    total_episodes = EPISODES_PER_PAIR * len(pair_envs)

    for ep in range(1, total_episodes + 1):
        # 通貨ペアをランダムに選択
        env = np.random.choice(pair_envs)

        (
            states, actions, rewards, dones,
            old_log_probs, values, last_value
        ) = collect_trajectory(env, model, STEPS_PER_EP)

        adv, ret = compute_gae(rewards, dones, values, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32,
                                       device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        dataset_size = states_t.size(0)
        total_reward = float(np.sum(rewards))
        episode_rewards.append(total_reward)

        # === PPO update ===
        for _ in range(EPOCHS_PPO):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, MINI_BATCH):
                end = start + MINI_BATCH
                mb_idx = idx[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logp = old_log_probs_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits, values_pred = model(mb_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(mb_actions)

                ratio = torch.exp(logp - mb_old_logp)
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

        print(f"[Hybrid PPO MultiAsset] Episode {ep}/{total_episodes} "
              f"steps={len(rewards)} reward={total_reward:.6f}")

        # 最後のエピソードのエクイティカーブ保存
        equity = 1.0
        curve = [equity]
        for r in rewards:
            equity *= (1.0 + r)
            curve.append(equity)
        last_equity_curve = curve

    # ==== 結果プロット ====
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards)
    plt.title("Hybrid PPO Multi-Asset Episode Rewards")
    plt.grid()
    plt.tight_layout()
    plt.savefig("hybrid_ppo_multiasset_episode_rewards.png")
    print("Saved: hybrid_ppo_multiasset_episode_rewards.png")

    if last_equity_curve is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(last_equity_curve)
        plt.title("Hybrid PPO Multi-Asset Trader Equity Curve (last episode)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("hybrid_ppo_multiasset_equity_curve.png")
        print("Saved: hybrid_ppo_multiasset_equity_curve.png")


if __name__ == "__main__":
    print("=== Hybrid LSTM+Transformer+Regime PPO (Multi-Asset) ===")
    train_hybrid_ppo_multi_asset()
