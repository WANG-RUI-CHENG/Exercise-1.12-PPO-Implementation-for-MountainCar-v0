"""
PPO for MountainCar-v0 (NO TensorBoard / NO tensorboardX)

- 相容 gym (<=0.25) 與 gymnasium / gym (>=0.26) 的 reset/step 回傳格式
- 不使用任何 SummaryWriter / tensorboardX
- 使用 log-prob 版本的 PPO ratio：ratio = exp(new_log_prob - old_log_prob)

執行：
    python PPO_MountainCar-v0_noTB.py
"""

import time
from collections import namedtuple
from itertools import count

import numpy as np

# 優先支援 gymnasium，沒有就用 gym
try:
    import gymnasium as gym
except Exception:
    import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


# ──────────────────────────────────────────────────────────────
# 參數
# ──────────────────────────────────────────────────────────────
ENV_NAME = "MountainCar-v0"
GAMMA = 0.99
SEED = 1
RENDER = False

LR_ACTOR = 1e-3
LR_CRITIC = 3e-3

# PPO 超參數
CLIP_PARAM = 0.2
MAX_GRAD_NORM = 0.5
PPO_UPDATE_TIME = 10
BATCH_SIZE = 32

# 訓練回合
MAX_EPISODES = 1000
PRINT_EVERY = 10

Transition = namedtuple(
    "Transition", ["state", "action", "logp", "reward", "next_state"]
)


# ──────────────────────────────────────────────────────────────
# Gym API 相容處理
# ──────────────────────────────────────────────────────────────
def reset_env(env, seed=None):
    """兼容 reset 回傳 obs 或 (obs, info)"""
    if seed is not None:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            # 舊 gym
            try:
                env.seed(seed)
            except Exception:
                pass
            out = env.reset()
    else:
        out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out


def step_env(env, action):
    """兼容 step 回傳 (obs, r, done, info) 或 (obs, r, terminated, truncated, info)"""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, r, done, info
    obs, r, done, info = out
    return obs, r, bool(done), info


def set_seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# 網路
# ──────────────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, num_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.action_head(x), dim=1)


class Critic(nn.Module):
    def __init__(self, num_state):
        super().__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.v(x)


# ──────────────────────────────────────────────────────────────
# PPO Agent
# ──────────────────────────────────────────────────────────────
class PPO:
    def __init__(self, num_state, num_action):
        self.actor = Actor(num_state, num_action)
        self.critic = Critic(num_state)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = []
        self.training_step = 0

    def select_action(self, state):
        s = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(s)
        dist = Categorical(probs)
        a = dist.sample()  # shape: [1]
        logp = dist.log_prob(a)  # shape: [1]
        return int(a.item()), float(logp.item())

    def store(self, trans):
        self.buffer.append(trans)

    def update(self, i_ep):
        # 整理 batch
        states = torch.tensor([t.state for t in self.buffer], dtype=torch.float32)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(
            -1, 1
        )
        rewards = [t.reward for t in self.buffer]
        old_logp = torch.tensor(
            [t.logp for t in self.buffer], dtype=torch.float32
        ).view(-1, 1)

        # discounted returns
        R = 0.0
        Gt = []
        for r in rewards[::-1]:
            R = r + GAMMA * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float32).view(-1, 1)

        # 進行多輪 PPO 更新
        n = len(self.buffer)
        for _ in range(PPO_UPDATE_TIME):
            for idx in BatchSampler(
                SubsetRandomSampler(range(n)), BATCH_SIZE, drop_last=False
            ):
                idx = list(idx)

                V = self.critic(states[idx])  # [B,1]
                advantage = (Gt[idx] - V).detach()  # [B,1]
                # （可選）讓 advantage 更穩
                if advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / (
                        advantage.std() + 1e-8
                    )

                # new log prob
                new_probs = self.actor(states[idx]).gather(1, actions[idx])  # [B,1]
                new_logp = torch.log(new_probs.clamp(min=1e-10))  # [B,1]

                # PPO ratio
                ratio = torch.exp(new_logp - old_logp[idx])  # [B,1]
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantage

                # actor update
                actor_loss = -torch.min(surr1, surr2).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optim.step()

                # critic update
                value_loss = F.mse_loss(V, Gt[idx])
                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optim.step()

                self.training_step += 1

        self.buffer.clear()


def main():
    set_seed_everywhere(SEED)

    env = gym.make(ENV_NAME)
    try:
        env = env.unwrapped
    except Exception:
        pass

    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n

    agent = PPO(num_state, num_action)

    for ep in range(1, MAX_EPISODES + 1):
        state = reset_env(env, seed=SEED + ep)
        if RENDER:
            env.render()

        ep_return = 0.0
        start_t = time.time()

        for t in count():
            action, logp = agent.select_action(state)
            next_state, reward, done, _ = step_env(env, action)

            if RENDER:
                env.render()

            agent.store(Transition(state, action, logp, reward, next_state))
            state = next_state
            ep_return += reward

            if done:
                # MountainCar 每回合最多 200 步，reward 幾乎都是 -1
                if len(agent.buffer) >= BATCH_SIZE:
                    agent.update(ep)

                if ep % PRINT_EVERY == 0:
                    dt = time.time() - start_t
                    print(
                        f"[Episode {ep:4d}] steps={t:3d}, return={ep_return:7.1f}, time={dt:5.2f}s"
                    )
                break

    env.close()
    print("end")


if __name__ == "__main__":
    main()
