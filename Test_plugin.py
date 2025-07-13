import socket
import numpy as np

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
                #if len(parts) != 7:
                #    print(f"[WARN] Ungültige Daten empfangen: {line}")
                #    continue

                x, y, speed, dist, yaw, pitch, cp = map(float, parts)
                cp = int(cp)

                # Checkpoint-Belohnung prüfen
                if cp > self.last_checkpoint:
                    print(f"✅ Neuer Checkpoint erreicht! ({cp}) → Belohnung")
                    self.last_checkpoint = cp

                return np.array([x, y, speed, dist, yaw, pitch, cp], dtype=np.float32)

        except Exception as e:
            print(f"[ERROR] Fehler beim Empfangen: {e}")
            return None


if __name__ == "__main__":
    client = TelemetryClient()
    try:
        while True:
            obs = client._get_obs()
            if obs is not None:
                print("[DATA]", ", ".join(f"{v:.6f}" for v in obs))
    except KeyboardInterrupt:
        print("\n[INFO] Programm abgebrochen durch Benutzer.")
    finally:
        client.sock.close()
        print("[INFO] Verbindung geschlossen.")
