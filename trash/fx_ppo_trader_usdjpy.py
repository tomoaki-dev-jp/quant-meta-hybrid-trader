# fx_ppo_trader_usdjpy.py
#
# TradingEnv（DDQNと同じ環境）を使った
# 雑だけど動くPPOサンプル（離散行動版）

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from fx_dqn_trader_usdjpy_ddqn import (
    load_close_series, TradingEnv, STATE_LEN, MAX_POSITION
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

EPISODES      = 100
STEPS_PER_EP  = 2000
N_ACTIONS     = 5       # [-2,-1,0,1,2]
GAMMA         = 0.99
LAMBDA_GAE    = 0.95
LR            = 3e-4
CLIP_EPS      = 0.2
EPOCHS_PPO    = 5
MINI_BATCH    = 1024

# ===== Actor-Critic ネットワーク =====

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
    """
    1エピソード分ロールアウトして軌跡を集める
    """
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

    # 最後の状態のvalue
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
        delta = rewards[t] + gamma * (last_value if t == T-1 else values[t+1]) * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def train_ppo():
    df = load_close_series()
    prices = df["close"].values
    env = TradingEnv(prices, state_len=STATE_LEN)

    state_dim = STATE_LEN + 3  # env側の仕様に合わせる
    model = ActorCritic(state_dim, N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = []
    equity_curves = []

    for ep in range(1, EPISODES + 1):
        # ===== rollout =====
        (
            states, actions, rewards, dones,
            old_log_probs, values, last_value
        ) = collect_trajectory(env, model, STEPS_PER_EP)

        adv, ret = compute_gae(rewards, dones, values, last_value)
        # 正規化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # tensor化
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        total_reward = np.sum(rewards)
        episode_rewards.append(total_reward)

        # ===== PPO update =====
        dataset_size = states_t.size(0)
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

        print(f"[PPO] Episode {ep}/{EPISODES}  "
              f"steps={len(rewards)}  reward={total_reward:.6f}")

        # ===== equity curve for monitoring =====
        # 同じログを equity に変換して保存
        equity = 1.0
        curve = [equity]
        for r in rewards:
            equity *= (1.0 + r)
            curve.append(equity)
        equity_curves.append(curve)

    # ===== 結果プロット =====
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards)
    plt.title("PPO Episode Rewards")
    plt.grid()
    plt.tight_layout()
    plt.savefig("ppo_episode_rewards.png")
    print("Saved: ppo_episode_rewards.png")

    # 最後のエピソードのエクイティカーブ
    if equity_curves:
        plt.figure(figsize=(10, 4))
        plt.plot(equity_curves[-1])
        plt.title("PPO Trader Equity Curve (last episode)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("ppo_equity_curve.png")
        print("Saved: ppo_equity_curve.png")


if __name__ == "__main__":
    print("PPO main start")
    train_ppo()
    print("PPO main end")
