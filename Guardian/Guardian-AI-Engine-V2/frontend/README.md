# Guardian AI Engine - Frontend Integration Guide

## Overview
Guardian AI Engine V2 has a real-time, event-driven architecture.
- **Analyze Engine**: The "Brain". It processes network traffic (live or simulated) in the background and detects attacks.
- **Redis**: The message bus. Adaptation and detection results are published here.
- **API Server**: The gateway. It connects to Redis and streams these alerts to the Frontend via WebSockets.

## Connection Details

### Base URL
Running locally: `http://localhost:8000`

### Real-Time Alerts (WebSocket)
**Endpoint:** `ws://localhost:8000/ws/alerts`

This is the primary data source for your dashboard. Connect to this WebSocket to receive live traffic analysis.

#### Data Format (JSON)
Every message received over the WebSocket will be a JSON object representing a processed packet/flow.

```json
{
  "timestamp": 1704283045.123,
  "source": "simulated_csv",
  "is_attack": true,
  "confidence": 0.9876,
  "attack_type": "Malicious",
  "original_features": {
    "Destination Port": 80,
    "Flow Duration": 1500,
    "Total Fwd Packets": 55,
    "..." : "..."
  }
}
```

- `is_attack`: `true` if the engine detected an anomaly/attack. `false` for normal traffic.
- `confidence`: Float between 0.0 and 1.0. Higher means more certain it's an attack.
- `attack_type`: Currently "Malicious" or "Benign". (Future versions may specify "DDoS", "PortScan", etc.)
- `original_features`: The raw network stats identifying the packet.

### Usage Example (JavaScript)

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/alerts');

socket.onopen = function(e) {
  console.log("[open] Connection established");
};

socket.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  if (data.is_attack) {
    // Trigger Red Alert UI
    updateDashboard(data, 'alert');
  } else {
    // Update "Processed Packets" counter or Live Traffic Graph
    updateDashboard(data, 'normal');
  }
};

socket.onclose = function(event) {
  if (event.wasClean) {
    console.log(`[close] Connection closed cleanly`);
  } else {
    // e.g. server process killed or network down
    // implement reconnect logic here
    console.log('[close] Connection died');
  }
};
```

## System Status Check
**Endpoint:** `GET /status`

Use this to show a "System Health" indicator in the UI.

**Response:**
```json
{
  "status": "online",
  "service": "Guardian AI Engine API",
  "role": "Server/Gateway",
  "redis": "connected"
}
```

## Workflow for Development
1. Start the backend services (Docker or manually).
   - This starts `redis`, `api`, and the `engine` (simulation mode).
2. Connect your frontend to the WebSocket.
3. You should immediately see JSON packets flowing in coming from the simulation engine.
