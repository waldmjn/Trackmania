Net::Socket@ serverSocket;
Net::Socket@ clientSocket;

int checkpointCount = 0;
int lastCheckpoint = -1;
uint lastCheckpointTime = Time::Now; // Zeittracking

void Main() {
    print("[PLUGIN] RL Interface Plugin gestartet.");

    @serverSocket = Net::Socket();
    serverSocket.Listen(1337);
    print("[PLUGIN] Lausche auf Port 1337...");

    startnew(UpdateLoop);
}

void UpdateLoop() {
    while (true) {
        yield();

        if (clientSocket is null) {
            @clientSocket = serverSocket.Accept();
            if (clientSocket !is null) {
                print("[PLUGIN] Client verbunden.");
            }
        }

        if (clientSocket !is null) {
            SendTelemetry();
            
        }
    }
}

void SendTelemetry() {
    auto scriptPlayer = GetScriptPlayer();
    if (scriptPlayer is null) return;
    
    auto player = GetPlayer();
    if (player is null) return;

    


    vec3 pos = scriptPlayer.Position;
    float speed = scriptPlayer.Speed;
    float distance = scriptPlayer.Distance;
    float yaw = scriptPlayer.AimYaw;
    float pitch = scriptPlayer.AimPitch;

    float dummyDeltaTime = -1.0;

    string msg = "" + pos.x + "," + pos.z + "," + speed + "," + distance + "," + yaw + "," + pitch;

    int currentCheckpoint = player.CurrentLaunchedRespawnLandmarkIndex;
    if (currentCheckpoint != lastCheckpoint) {
            lastCheckpoint = currentCheckpoint;
            checkpointCount++;
                uint now = Time::Now;
                float deltaTime = float(now - lastCheckpointTime) / 1000.0;
                lastCheckpointTime = now;

            msg = msg + "," + checkpointCount + "," + deltaTime + "\n";
    }
    else {
            msg = msg + "," + "-1" + "," + "-1" + "\n";
    }
    if (!clientSocket.Write(msg)) {
        print("[PLUGIN] Senden fehlgeschlagen. Trenne Client.");
        @clientSocket = null;
    }
}


// Hilfsfunktionen
CSmPlayer@ GetPlayer() {
    auto playground = cast<CSmArenaClient>(GetApp().CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0)
        return null;
    return cast<CSmPlayer>(playground.GameTerminals[0].GUIPlayer);
}

CSmScriptPlayer@ GetScriptPlayer() {
    auto player = GetPlayer();
    if (player is null) return null;
    return cast<CSmScriptPlayer>(player.ScriptAPI);
}
