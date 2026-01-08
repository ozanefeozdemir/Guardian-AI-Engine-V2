import os
import json
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import asyncio

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALERT_QUEUE = "alerts_queue"

# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect to Redis
    app.state.redis = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    try:
        await app.state.redis.ping()
        print(f"Connected to Redis at {REDIS_URL}")
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")
        
    # Start consumer task
    app.state.consumer_task = asyncio.create_task(redis_consumer())
        
    yield
    
    # Shutdown
    if hasattr(app.state, "consumer_task"):
        app.state.consumer_task.cancel()
    await app.state.redis.close()

app = FastAPI(title="Guardian AI Engine API", lifespan=lifespan)

# --- WebSocket Infrastructure ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Handle potential broken pipe if disconnect wasn't clean
                pass

manager = ConnectionManager()

# --- Background Consumer ---
async def redis_consumer():
    """
    Independent task to consume alerts from Redis Queue (BLPOP).
    This ensures no alerts are lost even if API is temporarily busy.
    """
    print(f"Starting Queue Consumer for: {ALERT_QUEUE}")
    
    # Create a dedicated connection for blocking operations
    redis_conn = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    
    try:
        while True:
            try:
                # BLPOP returns a tuple: (key, value) or None if timeout
                # It blocks until data is available.
                result = await redis_conn.blpop(ALERT_QUEUE, timeout=1)
                
                if result:
                    _, data = result
                    # Broadcast the JSON string directly to all connected clients
                    await manager.broadcast(data)
                
                # Small yield to prevent CPU hogging in case of very loop tight errors
                # (Though BLPOP handles waiting efficiently)
                await asyncio.sleep(0.001) 
                
            except asyncio.CancelledError:
                print("Consumer task cancelled.")
                break
            except Exception as e:
                print(f"Consumer Error: {e}")
                await asyncio.sleep(1) # Wait a bit before retrying on error
                
    finally:
        await redis_conn.close()

# --- Endpoints ---

@app.get("/status")
async def get_status():
    """Health check endpoint."""
    redis_status = "unknown"
    try:
        if await app.state.redis.ping():
            redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "online",
        "service": "Guardian AI Engine API",
        "role": "Server/Gateway",
        "redis": redis_status
    }

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open. We don't expect input from client, 
            # but we need to await something to keep the loop alive.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
