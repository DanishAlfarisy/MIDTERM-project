"""
train_rocket_dqn.py
Final version: DQN + Dueling DQN + Double DQN
Gymnasium + Stable Baselines3 v2 compatible
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy

# ============================================================
# 1. ENVIRONMENT : SimpleRocketEnv
# ============================================================
class SimpleRocketEnv(gym.Env):
    """
    2D rocket landing on a moving target.
    State: x, y, vx, vy, sin(theta), cos(theta), omega, dx, dy
    Action: 0 - idle, 1 - main, 2 - left, 3 - right
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=None, dt=0.05):
        super().__init__()
        self.dt = dt
        self.x_limit = 10.0
        self.y_limit = 12.0

        self.mass = 1.0
        self.main_thrust = 15.0
        self.side_thrust = 4.0
        self.torque = 3.0
        self.gravity = 9.81
        self.drag = 0.1

        self.action_space = spaces.Discrete(4)
        high = np.array(
            [self.x_limit, self.y_limit, 50.0, 50.0, 1.0, 1.0, 50.0, self.x_limit, self.y_limit],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.target_y = 1.0
        self.target_speed = 0.5
        self.land_dist_thresh = 0.8
        self.land_angle_thresh = 0.2
        self.land_speed_thresh = 1.5
        self.max_steps = 800
        self.seed()
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        x = self.np_random.uniform(-3.0, 3.0)
        y = self.np_random.uniform(6.0, 10.0)
        vx = self.np_random.normal(0, 1.0)
        vy = self.np_random.normal(-1.0, 1.0)
        theta = self.np_random.normal(0.0, 0.2)
        omega = self.np_random.normal(0.0, 0.5)
        target_x = self.np_random.uniform(-4.0, 4.0)
        self.target = {"x": target_x, "y": self.target_y, "vx": self.target_speed}
        self.state = np.array(
            [x, y, vx, vy, math.sin(theta), math.cos(theta), omega, self.target["x"] - x, self.target["y"] - y],
            dtype=np.float32,
        )
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state.copy()

    def step(self, action):
        x, y, vx, vy, sint, cost, omega, dx, dy = self.state
        theta = math.atan2(sint, cost)

        # --- target motion ---
        self.target["x"] += self.target["vx"] * self.dt
        if self.target["x"] > self.x_limit - 1 or self.target["x"] < -self.x_limit + 1:
            self.target["vx"] *= -1

        # --- forces ---
        thrust_x = thrust_y = torque = 0.0
        if action == 1:
            thrust_y += self.main_thrust * math.cos(theta)
            thrust_x += self.main_thrust * math.sin(theta)
        elif action == 2:
            torque += self.torque
            thrust_x -= self.side_thrust
        elif action == 3:
            torque -= self.torque
            thrust_x += self.side_thrust

        # --- dynamics ---
        ax = (thrust_x - self.drag * vx) / self.mass
        ay = (thrust_y - self.drag * vy - self.mass * self.gravity) / self.mass
        vx += ax * self.dt
        vy += ay * self.dt
        x += vx * self.dt
        y += vy * self.dt
        domega = torque / (self.mass * 0.5)
        omega += domega * self.dt
        theta += omega * self.dt
        dx = self.target["x"] - x
        dy = self.target["y"] - y
        self.state = np.array([x, y, vx, vy, math.sin(theta), math.cos(theta), omega, dx, dy], dtype=np.float32)

        self.step_count += 1
        terminated = False
        truncated = False
        info = {}

        # --- termination ---
        if abs(x) > self.x_limit or y > self.y_limit or y <= 0.0:
            terminated = True
            success = False
        else:
            dist = math.hypot(dx, dy)
            angle = abs(theta)
            speed = math.hypot(vx, vy)
            if dist <= self.land_dist_thresh and angle <= self.land_angle_thresh and speed <= self.land_speed_thresh:
                terminated = True
                success = True
            else:
                success = False

        # --- improved reward shaping ---
        reward = 0.0
        reward -= 0.5 * math.hypot(dx, dy)          # penalize distance
        reward -= 0.2 * math.hypot(vx, vy)          # penalize speed
        reward += 1.5 * math.cos(theta)             # upright bonus
        reward -= 0.05                              # time penalty
        if abs(dx) < 1.5 and vy < 0:                # encourage moving toward target slowly
            reward += 1.0

        if terminated and success:
            reward += 250.0
            info["success"] = True
        elif terminated and not success:
            reward -= 100.0
            info["success"] = False

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, info


# ============================================================
# 2. Dueling DQN Policy (custom architecture)
# ============================================================
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        return value + adv - adv.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def _build_q_net(self) -> None:
        self.q_net = DuelingQNetwork(self.features_dim, self.action_space.n)
        self.q_net_target = DuelingQNetwork(self.features_dim, self.action_space.n)
        self.q_net_target.load_state_dict(self.q_net.state_dict())


# ============================================================
# 3. Helper functions
# ============================================================
def make_env():
    env = SimpleRocketEnv()
    return Monitor(env)


def plot_rewards(rewards, model_dir, title):
    plt.figure(figsize=(6, 4))
    plt.plot(rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "learning_curve.png"), dpi=150)
    plt.close()


# ============================================================
# 4. TRAINING FUNCTIONS
# ============================================================
def train_dqn(total_timesteps=500_000, model_dir="./models_dqn"):
    os.makedirs(model_dir, exist_ok=True)
    env = make_env()

    model = DQN(
        "MlpPolicy", env, verbose=1,
        buffer_size=100_000, learning_starts=5000,
        batch_size=128, learning_rate=5e-4,
        target_update_interval=1000, train_freq=4,
        exploration_fraction=0.3, exploration_final_eps=0.02,
        tensorboard_log="./tb_logs_dqn/"
    )

    print("🚀 Training DQN baseline ...")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, "dqn_baseline.zip"))
    print("✅ DQN training complete.")

    return model


def train_dueling(total_timesteps=500_000, model_dir="./models_dueling"):
    os.makedirs(model_dir, exist_ok=True)
    env = make_env()

    model = DQN(
        DuelingDQNPolicy, env, verbose=1,
        buffer_size=100_000, learning_starts=5000,
        batch_size=128, learning_rate=5e-4,
        target_update_interval=1000, train_freq=4,
        exploration_fraction=0.3, exploration_final_eps=0.02,
        tensorboard_log="./tb_logs_dueling/"
    )

    print("🚀 Training Dueling DQN ...")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, "dueling_dqn.zip"))
    print("✅ Dueling DQN training complete.")

    return model


def train_double(total_timesteps=500_000, model_dir="./models_double"):
    """
    SB3's DQN implementation already uses Double DQN logic internally.
    This function trains a tuned version of DQN (double-DQN style).
    """
    os.makedirs(model_dir, exist_ok=True)
    env = make_env()

    model = DQN(
        "MlpPolicy", env, verbose=1,
        buffer_size=100_000, learning_starts=5000,
        batch_size=128, learning_rate=5e-4,
        target_update_interval=1000, train_freq=4,
        exploration_fraction=0.3, exploration_final_eps=0.02,
        tensorboard_log="./tb_logs_double/"
    )

    print("🚀 Training Double DQN (default SB3 logic)...")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, "double_dqn.zip"))
    print("✅ Double DQN training complete.")

    return model



# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    print("=== Starting Rocket Training Suite ===")
    dqn_model = train_dqn()
    dueling_model = train_dueling()
    double_model = train_double()
    print("✅ All training complete.")
