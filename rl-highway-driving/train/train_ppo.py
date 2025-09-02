import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from utils.logger import Logger
import os
import imageio

# مسیر ذخیره
log_dir = "logs/ppo"
video_dir = "videos/ppo"
os.makedirs(video_dir, exist_ok=True)

# پیکربندی محیط
config = {
    "observation": {"type": "Kinematics"},
    "policy_frequency": 2,
    "duration": 40
}

env = gym.make("highway-v0", render_mode="rgb_array", config=config)

# مدل PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    tensorboard_log=log_dir
)

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
        if reward < -1:
            collisions += 1

    logger.log(ep, total_reward, steps, collisions, actions)

logger.close()
model.save(os.path.join(log_dir, "ppo_highway_model"))

# ذخیره ویدیو
frames = []
obs, _ = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break

video_path = os.path.join(video_dir, "ppo_demo.mp4")
imageio.mimsave(video_path, frames, fps=30)
env.close()
