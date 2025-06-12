import gym
import numpy as np
import socket
import time
import vgamepad as vg
from gym import spaces


class TrackmaniaEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=1337):
        super().__init__()

        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        # Observation space: [x, z, speed, distance]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Socket connection
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.settimeout(1.0)

        # Gamepad init
        self.gamepad = vg.VX360Gamepad()

        # Episode tracking
        self.prev_dist = 0.0
        self.timeout_counter = 0
        self.max_no_progress_steps = 20  # ~2 seconds

        # Sticky throttle state
        self.last_throttle = 1.0
        self.hold_throttle_for = 0

    def reset(self):
        self.prev_dist = 0.0
        self.timeout_counter = 0
        self.last_throttle = 1.0
        self.hold_throttle_for = 0

        # Reset controller
        self.gamepad.reset()
        self.gamepad.update()

        time.sleep(1.0)

        obs = self._get_obs()
        return obs if obs is not None else np.zeros(4, dtype=np.float32)

    def step(self, action):
        steer, throttle = action
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # Sticky throttle logic
        if throttle > 0.2:
            self.last_throttle = throttle
            self.hold_throttle_for = 3  # hold for 3 frames (~50ms)
        elif self.hold_throttle_for > 0:
            throttle = self.last_throttle
            self.hold_throttle_for -= 1
        else:
            self.last_throttle = 0.0

        # Gamepad Steuerung
        self.gamepad.left_joystick(x_value=int(steer * 32767), y_value=170)
        self.gamepad.right_trigger(value=int(throttle * 255))
        self.gamepad.update()

        time.sleep(1 / 60)

        obs = self._get_obs()
        if obs is None:
            return np.zeros(4, dtype=np.float32), -1.0, True, {}

        x, z, speed, dist = obs
        delta_dist = dist - self.prev_dist
        self.prev_dist = dist

        # Reward + Strafe bei Stillstand
        reward = delta_dist * 0.1 + speed * 0.001
        if delta_dist < 0.05 and speed < 1.0:
            reward -= 0.1

        # Fortschritts-Timeout
        if delta_dist < 0.05:
            self.timeout_counter += 1
        else:
            self.timeout_counter = 0

        done = False
        if self.timeout_counter >= self.max_no_progress_steps:
            done = True
            reward -= 1.0

        print(f"[STEP] steer={steer:.2f}, throttle={throttle:.2f}, Δdist={delta_dist:.3f}, speed={speed:.2f}, reward={reward:.4f}")
        return obs, reward, done, {}

    def _get_obs(self):
        try:
            data = self.sock.recv(1024)
            decoded = data.decode("utf-8", errors="ignore").strip()

            parts = decoded.split(",")
            if len(parts) != 4:
                print(f"[WARN] Invalid data: {decoded}")
                return None

            x, z, speed, dist = map(float, parts)
            return np.array([x, z, speed, dist], dtype=np.float32)

        except Exception as e:
            print(f"[ERROR] Failed to receive observation: {e}")
            return None

    def close(self):
        self.gamepad.reset()
        self.gamepad.update()
        self.sock.close()
