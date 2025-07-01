import time
import socket
import numpy as np
import vgamepad as vg
import gymnasium as gym
from gymnasium import spaces
import re

class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super(TrackmaniaEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.gamepad = vg.VX360Gamepad()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", 1337))
        self.sock.setblocking(True)

        self.prev_dist = 0.0
        self.prev_speed = 0.0
        self.timeout_counter = 0
        self.max_no_progress_steps = 180
        self.obs_fail_counter = 0
        self.max_obs_failures = 5
        self.left_turn_counter = 0
        self.min_grace_period = 5.0

        self.last_movement_time = time.time()
        self.reset_penalty = False
        self.episode_start_time = time.time()

        self.min_speed_threshold = 0.83
        self.standstill_time_threshold = 6.0

        self.prev_yaw = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self._recv_buffer = ""

    def _normalize_obs(self, x, z, speed, dist, yaw, pitch):
        return np.array([
            x / 1000.0,
            z / 1000.0,
            speed / 10.0,
            dist / 1000.0,
            yaw / np.pi,
            pitch / np.pi
        ], dtype=np.float32)

    def _get_obs(self):
        try:
            data = self.sock.recv(4096)
            decoded = data.decode("utf-8", errors="ignore")
            self._recv_buffer += decoded

            lines = self._recv_buffer.split("\n")
            self._recv_buffer = lines[-1]  # letzte (vermutlich unvollständige) Zeile behalten

            for line in reversed(lines[:-1]):  # nur vollständige Zeilen prüfen
                line = line.strip()
                if line.count(",") != 5:
                    continue  # keine gültige Observation (nicht genau 6 Werte)

                try:
                    x, z, speed, dist, yaw, pitch = map(float, line.split(","))

                    # Plausibilitätsprüfung: Nur echte Plugin-Werte akzeptieren
                    if not (0 <= x <= 1000 and 0 <= z <= 1000 and -np.pi <= yaw <= np.pi):
                        print(f"[SKIP] Unplausible Werte erkannt: x={x:.2f}, z={z:.2f}, yaw={yaw:.3f}")
                        continue

                    print(f"[OBS VALID] {line}")
                    return self._normalize_obs(x, z, speed, dist, yaw, pitch)
                except ValueError:
                    print(f"[PARSE FAIL] Ungültige Zahlen in: {line}")
                    continue

            print(f"[WARN] Keine gültige Observation in diesem Datenpaket:\n{decoded.strip()}")
            return None

        except Exception as e:
            print(f"[ERROR] Fehler beim Empfang der Observation: {e}")
            return None


    def _press_reset(self):
        print("[RESET] Reset-Button wird gedrückt...")
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.right_trigger(value=0)
        self.gamepad.update()
        time.sleep(0.1)

        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()
        time.sleep(0.6)  # länger halten
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self.gamepad.update()
        print("[RESET] Reset abgeschlossen.")

    def _kickstart(self, duration=1.0, throttle=1.0):
        self.gamepad.left_joystick(x_value=0, y_value=170)
        self.gamepad.right_trigger(value=int(throttle * 255))
        self.gamepad.update()
        print(f"[KICKSTART] Kickstart für {duration:.2f}s mit Throttle={throttle:.2f}")
        time.sleep(duration)
        self.gamepad.right_trigger(value=0)
        self.gamepad.update()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("[ENV] Reset wird durchgeführt...")

        self.prev_dist = 0.0
        self.prev_speed = 0.0
        self.timeout_counter = 0
        self.obs_fail_counter = 0
        self.left_turn_counter = 0
        self.last_movement_time = time.time()
        self.reset_penalty = False
        self.episode_start_time = time.time()

        self._press_reset()
        time.sleep(1.0)
        self._kickstart()

        for i in range(5):
            steer = np.random.uniform(-0.2, 0.2)
            throttle = 1.0
            self.gamepad.left_joystick(x_value=int(steer * 32767), y_value=170)
            self.gamepad.right_trigger(value=int(throttle * 255))
            self.gamepad.update()
            time.sleep(1 / 60)
        print("[ENV] Kickstart abgeschlossen. Warten auf erste gültige Observation...")

        obs = None
        for _ in range(10):
            obs = self._get_obs()
            if obs is not None:
                break
            time.sleep(0.1)

        final_obs = obs if obs is not None else np.zeros(6, dtype=np.float32)
        print("[ENV] Reset fertig.")
        return final_obs, {}

    def step(self, action):
        steer, throttle = action
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        self.prev_steer = 0.9 * self.prev_steer + 0.1 * steer
        self.prev_throttle = 0.9 * self.prev_throttle + 0.1 * throttle

        self.gamepad.left_joystick(x_value=int(self.prev_steer * 32767), y_value=170)
        self.gamepad.right_trigger(value=int(self.prev_throttle * 255))
        self.gamepad.update()
        time.sleep(1 / 60)

        obs = self._get_obs()
        if obs is None:
            self.obs_fail_counter += 1
            print(f"[OBS FAIL] Observation fehlgeschlagen ({self.obs_fail_counter}/{self.max_obs_failures})")
            if time.time() - self.episode_start_time <= self.min_grace_period:
                return np.zeros(6, dtype=np.float32), 0.0, False, False, {}
            elif self.obs_fail_counter < self.max_obs_failures:
                return np.zeros(6, dtype=np.float32), 0.0, False, False, {}
            else:
                print("[OBS FAIL] Zu viele fehlgeschlagene Observations. Episode wird beendet.")
                return np.zeros(6, dtype=np.float32), -1.0, True, True, {"reset_reason": "obs_fail"}

        self.obs_fail_counter = 0

        x, z, speed, dist, yaw, pitch = obs
        delta_dist = dist - self.prev_dist
        self.prev_dist = dist

        acceleration = speed - self.prev_speed
        self.prev_speed = speed

        current_time = time.time()

        if speed > self.min_speed_threshold or delta_dist > 0.01:
            print(f"[MOVEMENT] Speed={speed:.3f}, ΔDist={delta_dist:.4f} → Bewegung erkannt.")
            self.last_movement_time = current_time
            self.reset_penalty = False

        if current_time - self.episode_start_time <= self.min_grace_period:
            self.reset_penalty = False
        elif current_time - self.last_movement_time > self.standstill_time_threshold:
            print(f"[RESET] Kein Fortschritt seit {current_time - self.last_movement_time:.2f}s → Reset wird ausgelöst.")
            self._press_reset()
            self.last_movement_time = current_time
            self.reset_penalty = True
        else:
            self.reset_penalty = False

        yaw_delta = abs(yaw - self.prev_yaw)
        self.prev_yaw = yaw

        reward = delta_dist

        if delta_dist < 0:
            reward -= abs(delta_dist) * 2.0

        if self.reset_penalty:
            penalty = 1.0 if speed < 0.05 and delta_dist < 0.001 else 0.3
            reward -= penalty
            print(f"[REWARD] Reset-Penalty angewendet: -{penalty:.2f}")

        if speed < 0.3 and delta_dist < 0.005:
            self.timeout_counter += 1
            print(f"[TIMEOUT] Keine Bewegung: timeout_counter = {self.timeout_counter}")
        else:
            self.timeout_counter = 0

        terminated = False
        truncated = self.timeout_counter >= self.max_no_progress_steps

        if truncated:
            print("[TRUNCATED] Max. timeout_counter erreicht → Episode wird abgebrochen.")

        return obs, reward, terminated, truncated, {
            "reset": self.reset_penalty,
            "speed": speed,
            "dist": dist,
            "yaw_delta": yaw_delta
        }
