import os
import json
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import init_db, get_db, AsyncSessionLocal
from models import Alert

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/dbname")
ALERT_QUEUE = "alerts_queue"

# --- Database Setup ---
# (Relies on imports from database.py and models.py)


# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 0. Initialize DB Tables
    await init_db()
    
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
    Persists to Postgres and Broadcasts to WebSockets.
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
                    # 1. Broadcast to WebSocket clients
                    await manager.broadcast(data)
                    
                    # 2. Persist to Postgres
                    try:
                        alert_data = json.loads(data)
                        async with AsyncSessionLocal() as session:
                            new_alert = Alert(
                                timestamp=alert_data.get("timestamp"),
                                source=str(alert_data.get("source")),
                                is_attack=bool(alert_data.get("is_attack")),
                                confidence=float(alert_data.get("confidence", 0.0)),
                                attack_type=str(alert_data.get("attack_type", "Unknown")),
                                original_features=alert_data.get("original_features", {})
                            )
                            session.add(new_alert)
                            await session.commit()
                    except Exception as e:
                        print(f"DB Save Error: {e}")

                    # 3. Update Stats (Redis for Speed)
                    try:
                        alert_json = json.loads(data)
                        await redis_conn.incr("stats:total_packets")
                        
                        if alert_json.get("is_attack"):
                            await redis_conn.incr("stats:total_attacks")
                            a_type = alert_json.get("attack_type", "Unknown")
                            await redis_conn.incr(f"stats:attack_type:{a_type}")
                    except Exception as e:
                        print(f"Stats Error: {e}")
                
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
        "redis": redis_status,
        "database": "postgres"
    }

from sqlalchemy import select # Added for get_alerts
@app.get("/api/alerts")
async def get_alerts(limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Get historical alerts from Postgres."""
    try:
        stmt = select(Alert).order_by(Alert.id.desc()).limit(limit)
        result = await db.execute(stmt)
        alerts = result.scalars().all()
        return [
            {
                "id": a.id,
                "timestamp": a.timestamp,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "source": a.source,
                "is_attack": a.is_attack,
                "confidence": a.confidence,
                "attack_type": a.attack_type,
                "original_features": a.original_features,
            }
            for a in alerts
        ]
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/live_feed")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
