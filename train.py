from stable_baselines3 import SAC
from trackmania_env import TrackmaniaEnv
from stable_baselines3.common.callbacks import EvalCallback
from datetime import datetime

env = TrackmaniaEnv()

# Optional: Evaluation während des Trainings
eval_callback = EvalCallback(env, best_model_save_path='./models_sac/best/',
                             log_path='./logs_sac/', eval_freq=10000,
                             deterministic=True, render=False)

# SAC-Setup
model = SAC("MlpPolicy", env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",  # wichtige Komponente für gute Exploration
            train_freq=(1, "step"),
            gradient_steps=1,
            target_update_interval=1,
            tensorboard_log="./sac_trackmania_tensorboard/")

# Training
model.learn(total_timesteps=4_000_000, callback=eval_callback)

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"models_sac/sac_trackmania_{timestamp}")
