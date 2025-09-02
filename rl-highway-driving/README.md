# RL Highway Driving 🚗💨

This project implements **Reinforcement Learning algorithms (DQN & PPO)**
for autonomous driving tasks using the `highway-env` environment.

## 📂 Project Structure
- `train/train_dqn.py` → Train & evaluate DQN
- `train/train_ppo.py` → Train & evaluate PPO
- `utils/logger.py` → Save rewards, episode length, collisions, and actions to CSV
- `utils/plotter.py` → Plot results (reward curves, episode length, collisions, action distribution)
- `logs/` → Training logs and CSV files
- `videos/` → Rendered videos of trained models

## 🚀 How to Run
```bash
pip install -r requirements.txt
python train/train_dqn.py
python train/train_ppo.py
