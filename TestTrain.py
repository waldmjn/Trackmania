from Gym_env import TrackmaniaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Optional: Wrapper-Funktion für VectorEnv
def make_env():
    return TrackmaniaEnv()

# Vektorisiertes Environment – 1 Env reicht zum Start
env = make_vec_env(make_env, n_envs=1)

# PPO-Modell initialisieren
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_trackmania_tensorboard/"
)

# Training starten
model.learn(total_timesteps=4_000_000)

# Modell speichern
model.save("ppo_trackmania")

env.close()
