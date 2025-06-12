from stable_baselines3 import PPO
from trackmania_env import TrackmaniaEnv
from datetime import datetime


env = TrackmaniaEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"models/ppo_trackmania_{timestamp}")
