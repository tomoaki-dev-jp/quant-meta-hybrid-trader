# fx_dqn_trader_usdjpy_ddqn.py
#
# 高度版トレードDQN:
# - 状態: 直近リターン + ボラ + トレンド + 現ポジション
# - 行動: 5種類のポジション [-2, -1, 0, +1, +2]
# - 報酬: PnL - 取引コスト（損失は強調）
# - Double DQN で安定化

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

print("fx_dqn_trader_usdjpy_ddqn imported, __name__ =", __name__)

# ====== 設定 ======
CSV_FILE      = "fx/yf_USDJPYX_5m_max.csv"
USE_RESAMPLE  = False
RESAMPLE_RULE = "5min"
MAX_POINTS    = 20000

STATE_LEN     = 48      # 直近 48 本 ≒ 4時間
EPISODES      = 150
STEPS_PER_EP  = 3000

GAMMA         = 0.99
LR            = 1e-4

EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.995

TRANSACTION_COST = 0.00003   # ポジション変化1あたりのコスト（約0.3pipsイメージ）
LOSS_FACTOR      = 2.0       # 損失強調係数

BATCH_SIZE    = 256
MEMORY_SIZE   = 100000
TARGET_UPDATE = 5            # 5エピソードごとにターゲット更新

MAX_POSITION  = 2            # -2 ~ +2 の5段階

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print("Using device:", device)


# ====== データ読み込み（closeだけ使う） ======

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


# ====== トレード環境 ======

class TradingEnv:
    """
    状態:
      - 過去 STATE_LEN 本のリターン
      - 近1時間ボラ（returnsのstd）
      - 近3時間トレンド（リターン平均）
      - 現在ポジション（-2〜+2 を [-1,1] にスケーリング）
    行動:
      0..4 -> 絶対ポジション [-2,-1,0,+1,+2]
    """
    def __init__(self, prices, state_len=STATE_LEN):
        self.prices = prices.astype(np.float32)
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.state_len = state_len
        self.reset_idx = state_len  # 初期t
        self.max_t = len(self.returns) - 1
        self.position = 0  # -2..+2
        self.t = None

        print("[TradingEnv] init:",
              f"len(prices)={len(self.prices)}, len(returns)={len(self.returns)}")

    def reset(self):
        self.t = self.reset_idx
        self.position = 0
        return self._get_state()

    def _get_state(self):
        start = self.t - self.state_len
        ret_window = self.returns[start:self.t]

        # ボラ（近1時間 ≒ 12本）
        vol = np.std(self.returns[max(0, self.t-12):self.t])
        # トレンド（近3時間 ≒ 36本の平均）
        trend = np.mean(self.returns[max(0, self.t-36):self.t])

        pos_scaled = self.position / MAX_POSITION  # -1 ~ +1 にスケール

        state = np.concatenate(
            [ret_window,
             np.array([vol, trend, pos_scaled], dtype=np.float32)]
        )
        return state.astype(np.float32)

    def step(self, action_idx):
        """
        action_idx: 0..4 -> position in [-2, -1, 0, 1, 2]
        """
        action_to_pos = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
        new_pos = int(action_to_pos[action_idx])

        prev_pos = self.position
        self.position = new_pos

        pos_change = abs(self.position - prev_pos)

        r = self.returns[self.t]  # t→t+1のリターン
        pnl = self.position * r
        cost = pos_change * TRANSACTION_COST
        reward = pnl - cost

        # 損失を強調
        if reward < 0:
            reward *= LOSS_FACTOR

        self.t += 1
        done = self.t >= self.max_t
        next_state = self._get_state() if not done else None

        return next_state, float(reward), done, {}


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


# ====== Double DQNネットワーク ======

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

def train_ddqn():
    print("train_ddqn() start")

    df = load_close_series()
    prices = df["close"].values
    env = TradingEnv(prices, state_len=STATE_LEN)

    # state = STATE_LEN リターン + vol + trend + pos_scaled
    state_dim = STATE_LEN + 3
    n_actions = 5  # -2,-1,0,1,2

    print(f"[train_ddqn] state_dim={state_dim}, n_actions={n_actions}")

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPS_START

    episode_rewards = []

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            steps += 1

            # epsilon-greedy行動
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32,
                                       device=device).unsqueeze(0)
                    q_values = policy_net(s_t)
                    action_idx = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done, _ = env.step(action_idx)
            total_reward += reward

            memory.push(
                state, action_idx, reward,
                None if next_state is None else next_state,
                done,
            )

            state = next_state
            if done or steps >= STEPS_PER_EP:
                break

            # ---- 学習ステップ ----
            if len(memory) >= BATCH_SIZE:
                (
                    states_b, actions_b, rewards_b,
                    next_states_b, dones_b
                ) = memory.sample(BATCH_SIZE)

                # Q(s,a)
                q_vals = policy_net(states_b).gather(
                    1, actions_b.unsqueeze(1)
                ).squeeze(1)

                # Double DQN: 次の行動は policy_net、値は target_net
                with torch.no_grad():
                    next_q_policy = policy_net(next_states_b)
                    next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
                    next_q_target = target_net(next_states_b).gather(
                        1, next_actions
                    ).squeeze(1)
                    target = rewards_b + GAMMA * next_q_target * (1 - dones_b)

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # epsilon減衰
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # ターゲットネット更新
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)

        print(f"Episode {ep}/{EPISODES}  steps={steps}  "
              f"reward={total_reward:.6f}  epsilon={epsilon:.3f}")

    # ====== モデル保存 ======
    torch.save(policy_net.state_dict(), "ddqn_trader.pth")
    print("Saved: ddqn_trader.pth")

    # ====== 学習経過の報酬プロット ======
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards)
    plt.title("DDQN Episode Rewards")
    plt.grid()
    plt.tight_layout()
    plt.savefig("ddqn_episode_rewards.png")
    print("Saved: ddqn_episode_rewards.png")

    # ====== 学習済みモデルで1周バックテスト ======
    eval_env = TradingEnv(prices, state_len=STATE_LEN)
    state = eval_env.reset()
    equity = 1.0
    eq_curve = [equity]
    while True:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32,
                               device=device).unsqueeze(0)
            q_values = policy_net(s_t)
            action_idx = int(torch.argmax(q_values, dim=1).item())
        next_state, reward, done, _ = eval_env.step(action_idx)
        equity *= (1.0 + reward)
        eq_curve.append(equity)
        state = next_state
        if done:
            break

    plt.figure(figsize=(10, 4))
    plt.plot(eq_curve)
    plt.title("DDQN Trader Equity Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig("ddqn_equity_curve.png")
    print("Saved: ddqn_equity_curve.png")

    print("train_ddqn() end")


if __name__ == "__main__":
    print("main block start (DDQN)")
    train_ddqn()
    print("main block end (DDQN)")
