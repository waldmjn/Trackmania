import os
import csv
from datetime import datetime
from Gym_env import TrackmaniaEnv
from stable_baselines3 import PPO
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
            model_path = os.path.join(self.save_path, f"ppo_trackmania_{timestamp}")
            self.model.save(model_path)
            if self.verbose:
                print(f"Modell gespeichert: {model_path}")
        return True

# Callback zum Loggen der Loss-Werte
class LossLoggingCallback(BaseCallback):
    def __init__(self, log_path="loss_log.txt", verbose=0, tensorboard_log="./ppo_trackmania_tensorboard/"):
        super().__init__(verbose)
        self.log_path = log_path
        self.header_written = False
        self.writer = SummaryWriter(tensorboard_log)  # Initialize TensorBoard writer

    def _on_step(self) -> bool:
        # Überprüfen, ob die EpInfo-Daten vorhanden sind und Loss-Werte extrahiert werden können
        log_dict = self.model.logger.name_to_value

        # Loss-Werte extrahieren
        policy_loss = log_dict.get("train/loss")
        value_loss = log_dict.get("train/value_loss")
        entropy_loss = log_dict.get("train/entropy_loss")
        approx_kl = log_dict.get("train/approx_kl")
        
        if policy_loss is not None:
            # Die Loss-Werte in die Textdatei schreiben
            with open(self.log_path, "a") as logfile:
                if not self.header_written:
                    logfile.write("step\tpolicy_loss\tvalue_loss\tentropy_loss\tapprox_kl\n")
                    self.header_written = True
                logfile.write(f"{self.num_timesteps}\t{policy_loss}\t{value_loss}\t{entropy_loss}\t{approx_kl}\n")

            # Zusätzlich TensorBoard-Logging
            self.writer.add_scalar("Loss/Policy_Loss", policy_loss, self.num_timesteps)
            self.writer.add_scalar("Loss/Value_Loss", value_loss, self.num_timesteps)
            self.writer.add_scalar("Loss/Entropy_Loss", entropy_loss, self.num_timesteps)
            self.writer.add_scalar("Loss/Approx_KL", approx_kl, self.num_timesteps)

        return True

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
    n_steps=1024,               # Etwas weniger Schritte pro Update (damit das Modell schneller lernt)
    batch_size=128,             # Größere Batches für stabilere Updates
    n_epochs=20,                # Mehr Epochen für intensivere Anpassung an die Strecke
    gamma=0.99,                 # Discount-Faktor bleibt gleich
    gae_lambda=0.95,            # GAE bleibt gleich, da es gut funktioniert
    ent_coef=0.005,             # Weniger Exploration, um Overfitting zu begünstigen
    vf_coef=0.5,                # Wertfunktion bleibt gleich, keine Veränderung nötig
    max_grad_norm=0.5,          # Begrenzung der Gradienten bleibt gleich, stabiler
    learning_rate=1e-4,         # Lernrate weiter gesenkt für langsameres Lernen
    clip_range=0.2,             # Clip für PPO bleibt gleich, da der Standard gut funktioniert
    tensorboard_log="./ppo_trackmania_tensorboard/",  # TensorBoard Log-Pfad
    policy_kwargs=dict(net_arch=[128, 64]),  # Kleinere Netzwerkarchitektur für spezialisierte Lernfähigkeit
)

# Speichern alle (3 * n_steps) Schritte = alle 3 Epochs
save_callback = SaveEveryNEpochsCallback(
    save_freq=10 * model.n_steps,
    save_path="./saved_models5"
)

# Callback zum Loggen der Loss-Werte
loss_callback = LossLoggingCallback()

# Alle Callbacks zusammenstellen
callback = CallbackList([save_callback, loss_callback])

def redirect_stdout_to_log(log_file_path="output.txt"):
    # Öffne die Log-Datei im Anhängemodus (falls sie nicht existiert, wird sie erstellt)
    log_file = open(log_file_path, "a")
    
    # Umleiten der Standardausgabe (alles, was mit print() ausgegeben wird)
    sys.stdout = log_file

    print(f"Logging gestartet. Alle Ausgaben werden in {log_file_path} gespeichert.")

# Funktion zum Zurücksetzen der Standardausgabe
def reset_stdout():
    sys.stdout = sys.__stdout__  # Setzt stdout auf den ursprünglichen Wert zurück

redirect_stdout_to_log()

# Training mit Callback
model.learn(total_timesteps=10_000_000, callback=callback)

# Modell speichern
model.save("ppo_trackmania")

env.close()
