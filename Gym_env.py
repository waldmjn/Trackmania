import socket
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import vgamepad as vg

class TelemetryClient:
    def __init__(self, host='localhost', port=1337):
        self.server_address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.last_checkpoint = -1  # ← zum Vergleichen für Reward

        try:
            self.sock.connect(self.server_address)
            print("[INFO] Verbunden mit dem Server auf Port 1337.")
        except Exception as e:
            print(f"[ERROR] Verbindung zum Server fehlgeschlagen: {e}")
            exit(1)

    def _get_obs(self):
        try:
            data = self.sock.recv(1024)
            decoded = data.decode("utf-8", errors="ignore").strip()

            # Mehrere Zeilen möglich
            lines = decoded.split("\n")
            for line in lines:
                if not line.strip():
                    continue

                parts = line.strip().split(",")

                x, y, speed, dist, yaw, pitch, cp = map(float, parts)
                cp = int(cp)

                if cp > self.last_checkpoint:
                    print(f"✅ Neuer Checkpoint erreicht! ({cp}) → Belohnung")
                    cp_changed = 1
                    self.last_checkpoint = cp
                else:
                    cp_changed = 0

                return np.array([x, y, speed, dist, yaw, pitch, cp_changed], dtype= np.float32)


        except Exception as e:
            #print(f"[ERROR] Fehler beim Empfangen: {e}")
            return None


    def close(self):
        self.sock.close()
        print("[INFO] Verbindung zum Server geschlossen.")


class TrackmaniaEnv(gym.Env):
    metadata = {"render_modes": []}

    def append_float_to_txt(self,filename: str, value: float):
        try:
            with open(filename, "a") as f:
                f.write(f"{value}\n")
            print(f"[INFO] Wert {value} in {filename} gespeichert.")
        except Exception as e:
            print(f"[ERROR] Fehler beim Schreiben in Datei: {e}")




    def __init__(self):
        super(TrackmaniaEnv, self).__init__()
        self.last_reward = 0
        self.low_speed_start_time = None
        self.speed_threshold = 1.39  # 5 km/h in m/s
        self.low_speed_duration = 2.0  # Sekunden bis Reset

        self.action_space = spaces.Discrete(4)

        self.reward_sum = 0
       


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        self.client = TelemetryClient()
        self.current_obs = None

        try:
            self.gamepad = vg.VX360Gamepad()
        except Exception as e:
            print(f"[ERROR] Gamepad konnte nicht initialisiert werden: {e}")
            exit(1)

    def _get_valid_obs(self):
        while True:
            obs = self.client._get_obs()
            if obs is not None:
                return obs

    def _send_action_to_gamepad(self, action):
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_trigger_float(0.0)
        self.gamepad.left_trigger_float(0.0)

        if action == 0:
            self.gamepad.left_joystick_float(x_value_float=-1.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(1.0)
        elif action == 1:
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(1.0)
        elif action == 2:
            self.gamepad.left_joystick_float(x_value_float=1.0, y_value_float=0.0)
            self.gamepad.right_trigger_float(1.0)
        elif action == 3:
            self.gamepad.left_trigger_float(1.0)

        self.gamepad.update()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("[ENV] Resetting environment...")
        time.sleep(1)
        self.current_obs = self._get_valid_obs()
        return self.current_obs, {}

    def _perform_reset(self):
        print("[ENV] Reset wird über Gamepad ausgelöst (Taste B).")
        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()
        time.sleep(0.2)
        self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()
        self.last_checkpoint = -1

    def step(self, action):
        self._send_action_to_gamepad(action)
        obs = self._get_valid_obs()
        speed = obs[2]
        current_distance = obs[3]
        previous_distance = self.current_obs[3]
        driven_checkpoint = obs[6]
        self.current_obs = obs

        reward = 0.0
        reward += current_distance - previous_distance
        reward += speed * 0.15
        reward += 100 * driven_checkpoint
        
        if speed < self.speed_threshold:
    
            reward -= 150

        self.reward_sum += reward
        x, y = obs[0], obs[1]
        print(f"[LOG] Position: x={x:.3f}, y={y:.3f}, Speed={speed:.2f} m/s, Action={action}, Dist={current_distance:.2f}, Reward={reward:.4f}")
        
        #if reward > self.last_reward:
        #    print (reward)
        #    self.last_reward = reward

        now = time.time()
        if speed < self.speed_threshold:
            if self.low_speed_start_time is None:
                self.low_speed_start_time = now
            elif now - self.low_speed_start_time >= self.low_speed_duration:
                print("[INFO] Fahrzeug zu langsam – Reset über Taste B.")
                self._perform_reset()
                self.append_float_to_txt(filename="rewards.txt", value = self.reward_sum)
                self.reward_sum = 0
                return obs, reward, False, True, {"reason": "low_speed"}
            
        else:
            self.low_speed_start_time = None
        
        return obs, reward, False, False, {}
