# Reinforcement Learning mit Trackmania 2020

Dieses Projekt implementiert eine Reinforcement-Learning-Umgebung für Trackmania 2020 mithilfe von Python, Openplanet und vGamepad. Ziel ist es, einen Agenten zu trainieren, der auf einer benutzerdefinierten Strecke autonom fährt.

## 🛠 Voraussetzungen

Bevor das Projekt ausgeführt werden kann, müssen folgende Voraussetzungen erfüllt sein:

### 🕹 Spiel & Lizenz

- **Trackmania (2020)** muss installiert sein  
  → Verfügbar über Ubisoft Connect oder Epic Games Store
- **Club-Zugang erforderlich**  
  → Diese Lizenz ist notwendig, um benutzerdefinierte Strecken und Plugins (wie Openplanet) zu verwenden  
  → **Kosten:** ca. 20 €

### 🔌 Openplanet installieren

- Openplanet ist eine Modding- und Debugging-Plattform für Trackmania
- Download unter: [https://openplanet.dev](https://openplanet.dev)
- Nach der Installation muss das zugehörige **Trackmania-Plugin** installiert werden (Plugin für Telemetriedatenübertragung)

### 🐍 Python-Abhängigkeiten

Ein Python-Environment muss eingerichtet werden (z. B. mit venv oder conda). Die benötigten Pakete sind in der requirements.txt zu finden.

Ausführen der TestTrain.py startet das Training mit PPO und TestTrainDQN.py startet das Training mit DQN.
