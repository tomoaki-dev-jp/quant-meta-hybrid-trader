# fx_lstm_ppo_hybrid_trader.py
#
# LSTMベース予測 × テクニカル特徴 × 強化報酬 × 7段階ロット × PPO
# という全部盛りハイブリッドFXトレーダー。
#
# 手順:
#   1. LSTMで「次のリターン」を教師あり学習
#   2. 学習済みLSTMを、PPOエージェントの状態入力に埋め込む
#   3. TradingEnvで報酬を設計（損失強調・トレンド強調・コスト控除）
#   4. PPOでポリシー最適化
#
# 前提: CSVは yfinance から取得した USDJPY 5m データ
# カラムに "Price"（datetime）, "close" が存在する想定

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ================== グローバル設定 ==================

CSV_FILE      = "fx/yf_USDJPYX_5m_max.csv"
USE_RESAMPLE  = False
RESAMPLE_RULE = "5min"
MAX_POINTS    = 20000

# LSTM 用
LSTM_SEQ_LEN  = 64      # LSTMに入れる過去本数
LSTM_EPOCHS   = 30
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
LSTM_LR       = 1e-3
LSTM_BATCH    = 256

# RL 用
STATE_RET_LEN = 48      # PPO側の状態で使う過去リターン本数
EPISODES      = 80
STEPS_PER_EP  = 2000

GAMMA         = 0.99
LAMBDA_GAE    = 0.95
LR_PPO        = 3e-4
CLIP_EPS      = 0.2
EPOCHS_PPO    = 5
MINI_BATCH    = 1024

N_ACTIONS     = 7       # [-3,-2,-1,0,1,2,3]
MAX_POSITION  = 3

TRANSACTION_COST = 0.00003   # ポジション変化1あたりコスト
LOSS_FACTOR      = 2.0       # 損失の強調
TREND_THRESHOLD  = 0.0002    # トレンド強調の閾値
TREND_BOOST      = 1.5       # トレンド強調倍率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print("Using device:", device)

# ================== 共通：データ読み込み ==================

def load_close_series():
    print("[load_close_series] loading CSV:", CSV_FILE)
    df = pd.read_csv(CSV_FILE)

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
        print("[load_close_series] resampling:", RESAMPLE_RULE)
        df = df[["close"]].resample(RESAMPLE_RULE).last().dropna()

    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]
        print(f"[load_close_series] trimmed to last {MAX_POINTS} rows")

    print("[load_close_series] final rows:", len(df))
    return df

# ================== ステップ1: LSTM で次リターン予測 ==================

def build_lstm_dataset(prices: np.ndarray):
    """
    prices: 1D array of close
    戻り値:
      X: (N, LSTM_SEQ_LEN, input_dim)
      y: (N,) 次のリターン
    """
    returns = np.diff(prices) / prices[:-1]
    # テクニカル特徴
    r = returns
    # ボラ（12本）
    vol_12 = pd.Series(r).rolling(12).std().values
    # ボラ（36本）
    vol_36 = pd.Series(r).rolling(36).std().values
    # トレンド（36本平均）
    trend_36 = pd.Series(r).rolling(36).mean().values
    # RSI(14)
    r_series = pd.Series(r)
    up = r_series.clip(lower=0).rolling(14).mean()
    down = (-r_series.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 * up / (up + down + 1e-9)
    rsi = (rsi - 50.0) / 50.0   # -1〜+1くらいにスケール

    feat_mat = np.stack([
        r,
        np.nan_to_num(vol_12, nan=0.0),
        np.nan_to_num(vol_36, nan=0.0),
        np.nan_to_num(trend_36, nan=0.0),
        np.nan_to_num(rsi.values, nan=0.0)
    ], axis=1)  # (T-1, 5)

    # target: 1ステップ先リターン
    target = np.roll(r, -1)
    feat_mat = feat_mat[:-1]
    target = target[:-1]

    # LSTM用にシーケンス化
    X_list, y_list = [], []
    T = len(target)
    for t in range(LSTM_SEQ_LEN, T):
        X_list.append(feat_mat[t-LSTM_SEQ_LEN:t])
        y_list.append(target[t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print("[build_lstm_dataset] X:", X.shape, "y:", y.shape)
    return X, y


class LSTMReturnModel(nn.Module):
    def __init__(self, input_dim=5, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


def train_lstm(prices: np.ndarray):
    X, y = build_lstm_dataset(prices)
    N = len(X)
    train_size = int(N * 0.8)
    X_train = torch.tensor(X[:train_size], device=device)
    y_train = torch.tensor(y[:train_size], device=device)
    X_val   = torch.tensor(X[train_size:], device=device)
    y_val   = torch.tensor(y[train_size:], device=device)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=min(LSTM_BATCH, len(train_ds)),
        shuffle=True,
    )

    model = LSTMReturnModel(input_dim=X.shape[2]).to(device)
    opt = optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        total = 0
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
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        print(f"[LSTM] Epoch {epoch}/{LSTM_EPOCHS} "
              f"Train={train_loss:.6e} Val={val_loss:.6e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # 簡易評価
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).cpu().numpy()
        val_true = y_val.cpu().numpy()
    mse = np.mean((val_pred - val_true) ** 2)
    print(f"[LSTM] Val MSE={mse:.6e}")

    return model


def build_lstm_feature_feeder(prices: np.ndarray, lstm_model: LSTMReturnModel):
    """
    環境側から「今の時刻tまでのデータ」を渡したときに
    LSTMで次リターン予測を返す関数を作って返す。
    """
    closes = prices
    returns = np.diff(closes) / closes[:-1]
    r_series = pd.Series(returns)

    vol_12 = r_series.rolling(12).std().values
    vol_36 = r_series.rolling(36).std().values
    trend_36 = r_series.rolling(36).mean().values
    up = r_series.clip(lower=0).rolling(14).mean()
    down = (-r_series.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 * up / (up + down + 1e-9)
    rsi = (rsi - 50.0) / 50.0

    feat_mat = np.stack([
        returns,
        np.nan_to_num(vol_12, nan=0.0),
        np.nan_to_num(vol_36, nan=0.0),
        np.nan_to_num(trend_36, nan=0.0),
        np.nan_to_num(rsi.values, nan=0.0),
    ], axis=1)  # (T-1,5)

    def predict_next_return(t_index: int) -> float:
        """
        t_index: returnsのインデックスに対応（0〜len(returns)-1）
                 "t_index" の位置を現在とし、そこまでの履歴から t+1 のリターンを予測
        """
        if t_index < LSTM_SEQ_LEN:
            return 0.0
        end = t_index + 1
        start = end - LSTM_SEQ_LEN
        x = feat_mat[start:end]  # (LSTM_SEQ_LEN, 5)
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        lstm_model.eval()
        with torch.no_grad():
            pred = lstm_model(x_t).item()
        return float(pred)

    return predict_next_return


# ================== ステップ2: LSTM込みTradingEnv ==================

class HybridTradingEnv:
    """
    状態:
      - 過去 STATE_RET_LEN 本のリターン
      - ボラ (12本)
      - トレンド (36本)
      - RSI(14)
      - LSTM予測次リターン
      - 現在ポジション (スケーリング済)
    行動:
      0..6 -> position in [-3,-2,-1,0,1,2,3]
    """
    def __init__(self, prices, state_ret_len, lstm_predict_func):
        self.prices = prices.astype(np.float32)
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.state_ret_len = state_ret_len
        self.reset_idx = state_ret_len
        self.max_t = len(self.returns) - 1
        self.position = 0
        self.t = None

        self.r_series = pd.Series(self.returns)

        self.vol_12 = self.r_series.rolling(12).std().fillna(0.0).values
        self.trend_36 = self.r_series.rolling(36).mean().fillna(0.0).values
        up = self.r_series.clip(lower=0).rolling(14).mean()
        down = (-self.r_series.clip(upper=0)).rolling(14).mean()
        rsi = 100.0 * up / (up + down + 1e-9)
        self.rsi = ((rsi - 50.0) / 50.0).fillna(0.0).values

        self.lstm_predict_func = lstm_predict_func

        print("[HybridTradingEnv] init:",
              f"len(prices)={len(self.prices)}, len(returns)={len(self.returns)}")

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

        # LSTMで次リターン予測 (tをreturns indexとして渡す)
        lstm_pred = self.lstm_predict_func(self.t)

        pos_scaled = self.position / MAX_POSITION

        state = np.concatenate([
            ret_window,
            np.array([vol, trend, rsi_val, lstm_pred, pos_scaled], dtype=np.float32)
        ])
        return state.astype(np.float32)

    def step(self, action_idx):
        # アクション -> 絶対ポジション
        action_to_pos = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
        new_pos = int(action_to_pos[action_idx])

        prev_pos = self.position
        self.position = new_pos

        pos_change = abs(self.position - prev_pos)

        r = self.returns[self.t]
        pnl = self.position * r
        cost = pos_change * TRANSACTION_COST
        reward = pnl - cost

        # 損失強調
        if reward < 0:
            reward *= LOSS_FACTOR

        # トレンド強調：トレンドが強い時に報酬ブースト
        trend = self.trend_36[self.t]
        if abs(trend) > TREND_THRESHOLD:
            reward *= TREND_BOOST

        self.t += 1
        done = self.t >= self.max_t
        next_state = self._get_state() if not done else None

        return next_state, float(reward), done, {}


# ================== ステップ3: PPOエージェント ==================

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


def collect_trajectory(env, model, steps_per_ep):
    model.eval()
    state = env.reset()

    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    for _ in range(steps_per_ep):
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(s_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(int(action.item()))

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
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


def train_hybrid_ppo():
    # ---- データ準備 & LSTM学習 ----
    df = load_close_series()
    prices = df["close"].values

    print("\n=== Step 1: Train LSTM ===")
    lstm_model = train_lstm(prices)
    lstm_predict_func = build_lstm_feature_feeder(prices, lstm_model)

    print("\n=== Step 2: PPO with Hybrid Env ===")
    env = HybridTradingEnv(prices, STATE_RET_LEN, lstm_predict_func)

    # 状態次元: past returns + [vol, trend, rsi, lstm_pred, pos_scaled]
    state_dim = STATE_RET_LEN + 5
    model = ActorCritic(state_dim, N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_PPO)

    episode_rewards = []
    last_equity_curve = None

    for ep in range(1, EPISODES + 1):
        (
            states, actions, rewards, dones,
            old_log_probs, values, last_value
        ) = collect_trajectory(env, model, STEPS_PER_EP)

        adv, ret = compute_gae(rewards, dones, values, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        dataset_size = states_t.size(0)
        total_reward = float(np.sum(rewards))
        episode_rewards.append(total_reward)

        # ===== PPO update =====
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
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_pred, mb_ret)
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"[Hybrid PPO] Episode {ep}/{EPISODES} "
              f"steps={len(rewards)} reward={total_reward:.6f}")

        # エクイティカーブ（最後のエピソードだけ可視化用に保存）
        equity = 1.0
        curve = [equity]
        for r in rewards:
            equity *= (1.0 + r)
            curve.append(equity)
        last_equity_curve = curve

    # ===== 結果プロット =====
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards)
    plt.title("Hybrid PPO Episode Rewards (with LSTM forecasts)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("hybrid_ppo_episode_rewards.png")
    print("Saved: hybrid_ppo_episode_rewards.png")

    if last_equity_curve is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(last_equity_curve)
        plt.title("Hybrid PPO Trader Equity Curve (last episode)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("hybrid_ppo_equity_curve.png")
        print("Saved: hybrid_ppo_equity_curve.png")


if __name__ == "__main__":
    print("=== Hybrid LSTM + PPO Trader ===")
    train_hybrid_ppo()
