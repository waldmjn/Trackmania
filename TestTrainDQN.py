import os
import csv
from datetime import datetime
from Gym_env import TrackmaniaEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch.utils.tensorboard import SummaryWriter  # TensorBoard Logging
import sys

# Callback zum Speichern des Modells alle N-Epochen
class SaveEveryNEpochsCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.save_path, f"dqn_trackmania_{timestamp}")
            self.model.save(model_path)
            if self.verbose:
                print(f"Modell gespeichert: {model_path}")
        return True


# Callback zum Loggen der Loss-Werte
class LossLoggingCallback(BaseCallback):
    def __init__(self, log_path="loss_log.txt", verbose=0, tensorboard_log="./dqn_trackmania_tensorboard/"):
        super().__init__(verbose)
        self.log_path = log_path
        self.header_written = False
        self.writer = SummaryWriter(tensorboard_log)

    def _on_step(self) -> bool:
        log_dict = self.model.logger.name_to_value
        q_loss = log_dict.get("train/q_loss")  # DQN speichert den Q-Loss als "train/q_loss"
        
        if q_loss is not None:
            with open(self.log_path, "a") as logfile:
                if not self.header_written:
                    logfile.write("step\tq_loss\n")
                    self.header_written = True
                logfile.write(f"{self.num_timesteps}\t{q_loss}\n")
            self.writer.add_scalar("Loss/Q_Loss", q_loss, self.num_timesteps)

        return True

def redirect_stdout_to_log(log_file_path="output.txt"):
    log_file = open(log_file_path, "a")
    sys.stdout = log_file
    print(f"Logging gestartet. Alle Ausgaben werden in {log_file_path} gespeichert.")

def reset_stdout():
    sys.stdout = sys.__stdout__

redirect_stdout_to_log()

# Vektorisiertes Environment – 1 Env reicht zum Start
env = make_vec_env(lambda: TrackmaniaEnv(), n_envs=1)

# DQN-Modell initialisieren
model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

# Speichern alle (3 * n_steps) Schritte = alle 3 Epochs
save_callback = SaveEveryNEpochsCallback(
    save_freq=10000,
    save_path="./saved_models3"
)

# Callback zum Loggen der Loss-Werte
loss_callback = LossLoggingCallback()

# Alle Callbacks zusammenstellen
callback = CallbackList([save_callback, loss_callback])

# Training mit Callback
model.learn(total_timesteps=10_000_000, callback=callback)

# Modell speichern
model.save("dqn_trackmania")

env.close()
reset_stdout()
