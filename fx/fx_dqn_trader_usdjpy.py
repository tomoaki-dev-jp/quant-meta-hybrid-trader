# fx_dqn_trader_usdjpy.py
#
# USDJPY 5m のclose時系列から
# シンプルなDQNトレーダーを学習するサンプル。
# ・状態: 直近 N 本のリターン + 現ポジション
# ・行動: -1 (ショート), 0 (ノーポジ), +1 (ロング)
# ・報酬: ポジション × 価格変化 - 手数料

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ====== 設定 ======
CSV_FILE      = "fx/yf_USDJPYX_5m_max.csv"
USE_RESAMPLE  = False
RESAMPLE_RULE = "5min"
MAX_POINTS    = 20000

STATE_LEN     = 48      # 直近 48 本 ≒ 4時間
EPISODES      = 100
STEPS_PER_EP  = 2000    # 1エピソードで進むステップ数（データ長と相談）
GAMMA         = 0.99
LR            = 1e-4
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.995
TRANSACTION_COST = 0.00005   # 約0.5pips相当をイメージ

BATCH_SIZE    = 256
MEMORY_SIZE   = 50000
TARGET_UPDATE = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ====== データ読み込み（closeだけ使う） ======

def load_close_series():
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
        df = df[["close"]].resample(RESAMPLE_RULE).last().dropna()

    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]

    print("Loaded rows:", len(df))
    return df

# ====== トレード環境 ======

class TradingEnv:
    def __init__(self, prices, state_len=STATE_LEN):
        self.prices = prices.astype(np.float32)
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.state_len = state_len
        self.reset_idx = state_len  # 初期t
        self.max_t = len(self.returns) - 1
        self.position = 0  # -1,0,1
        self.t = None

    def reset(self):
        self.t = self.reset_idx
        self.position = 0
        return self._get_state()

    def _get_state(self):
        # 直近state_lenのリターン ＋ 現ポジション1要素
        start = self.t - self.state_len
        ret_window = self.returns[start:self.t]
        state = np.concatenate([ret_window, np.array([self.position], dtype=np.float32)])
        return state.astype(np.float32)

    def step(self, action):
        """
        action: -1,0,1 のいずれか
        """
        prev_pos = self.position
        self.position = int(action)

        # ポジション変化に応じて取引コスト
        pos_change = abs(self.position - prev_pos)

        # 現在のリターンに対してポジションを掛ける
        r = self.returns[self.t]  # t→t+1のリターン
        reward = self.position * r - pos_change * TRANSACTION_COST

        self.t += 1
        done = self.t >= self.max_t
        next_state = self._get_state() if not done else None

        return next_state, reward, done, {}

# ====== リプレイメモリ ======

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, ns, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, ns, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idx:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(np.zeros_like(s) if ns is None else ns)
            dones.append(d)
        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)

# ====== DQNネットワーク ======

class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ====== メイン学習ループ ======

def train_dqn():
    df = load_close_series()
    prices = df["close"].values
    env = TradingEnv(prices, state_len=STATE_LEN)

    state_dim = STATE_LEN + 1  # リターンwindow + 現ポジション
    n_actions = 3              # -1,0,1 を {0,1,2} にマップ

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPS_START

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            steps += 1
            # 行動選択
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s_t)
                    action_idx = int(torch.argmax(q_values, dim=1).item())

            action = action_idx - 1  # {0,1,2} -> {-1,0,1}

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            memory.push(
                state, action_idx, reward,
                None if next_state is None else next_state,
                done,
            )

            state = next_state
            if done or steps >= STEPS_PER_EP:
                break

            # 学習ステップ
            if len(memory) >= BATCH_SIZE:
                (
                    states_b, actions_b, rewards_b,
                    next_states_b, dones_b
                ) = memory.sample(BATCH_SIZE)

                q_vals = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states_b).max(1)[0]
                    target = rewards_b + GAMMA * next_q * (1 - dones_b)

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # epsilon減衰
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # ターゲットネット更新
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {ep}/{EPISODES}  steps={steps}  "
              f"reward={total_reward:.6f}  epsilon={epsilon:.3f}")

    # 学習済みモデル保存
    torch.save(policy_net.state_dict(), "dqn_trader.pth")
    print("Saved: dqn_trader.pth")

    # 簡易バックテスト（学習済みモデルで1周）
    eval_env = TradingEnv(prices, state_len=STATE_LEN)
    state = eval_env.reset()
    equity = 1.0
    eq_curve =_
