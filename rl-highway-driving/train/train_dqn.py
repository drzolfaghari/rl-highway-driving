import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from utils.logger import Logger
import numpy as np
import os
import imageio

# Config
log_dir = "logs/dqn"
video_dir = "videos/dqn"
os.makedirs(video_dir, exist_ok=True)

import highway_env

config = {
    "observation": {
        "type": "Kinematics"
    },
    "policy_frequency": 2,
    "duration": 40
}

env = gym.make("highway-v0", render_mode="rgb_array", config=config)

env.reset()

# Train model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=10000,
            batch_size=64, gamma=0.99, train_freq=1, target_update_interval=500,
            exploration_fraction=0.3, exploration_final_eps=0.05, tensorboard_log=log_dir)

logger = Logger(log_dir)

episodes = 100
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward, steps, collisions, actions = 0, 0, 0, 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        actions += action
        if reward < -1:  # simple collision detection
            collisions += 1

    logger.log(ep, total_reward, steps, collisions, actions)

logger.close()
model.save(os.path.join(log_dir, "dqn_highway_model"))

# Save video
frames = []
obs, _ = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break

video_path = os.path.join(video_dir, "dqn_demo.mp4")
imageio.mimsave(video_path, frames, fps=30)
env.close()
