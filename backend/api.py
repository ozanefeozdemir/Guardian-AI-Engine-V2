import os
import json
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import init_db, seed_data, get_db, AsyncSessionLocal # seed_data eklendi
from models import Alert, User, AuthLog
from auth import router as auth_router
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALERT_QUEUE = "alerts_queue"

# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 0. Initialize DB Tables (Tabloları kontrol et/oluştur)
    await init_db()
    
    # 1. Seed Default Data (Admin yoksa oluştur)
    await seed_data()
    
    # 2. Connect to Redis
    app.state.redis = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    try:
        await app.state.redis.ping()
        print(f"✅ Connected to Redis at {REDIS_URL}")
    except Exception as e:
        print(f"❌ Warning: Redis connection failed: {e}")
        
    # 3. Start consumer task (Arka plan dinleyicisini başlat)
    app.state.consumer_task = asyncio.create_task(redis_consumer())
        
    yield
    
    # Shutdown (Kapatırken temizlik yap)
    if hasattr(app.state, "consumer_task"):
        app.state.consumer_task.cancel()
    if hasattr(app.state, "redis"):
        await app.state.redis.close()

app = FastAPI(title="Guardian AI Engine API", lifespan=lifespan)

# --- Auth Rotalarını Bağlama ---
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])

# --- CORS Ayarları ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Infrastructure ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

# --- Background Consumer ---
async def redis_consumer():
    print(f"📡 Starting Queue Consumer for: {ALERT_QUEUE}")
    # Ayrı bir bağlantı üzerinden Redis dinlemesi yapılır
    redis_conn = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    
    try:
        while True:
            try:
                # Redis listesinden (queue) veri bekle
                result = await redis_conn.blpop(ALERT_QUEUE, timeout=1)
                if result:
                    _, data = result
                    # WebSocket üzerinden tüm Dashboard'lara bas
                    await manager.broadcast(data)
                    
                    # Veriyi parse et ve DB'ye kaydet
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
                        print(f"❌ DB Save Error: {e}")

                    # İstatistikleri güncelle
                    try:
                        await redis_conn.incr("stats:total_packets")
                        alert_json = json.loads(data)
                        if alert_json.get("is_attack"):
                            await redis_conn.incr("stats:total_attacks")
                            a_type = alert_json.get("attack_type", "Unknown")
                            await redis_conn.incr(f"stats:attack_type:{a_type}")
                    except Exception as e:
                        print(f"❌ Stats Error: {e}")
                
                await asyncio.sleep(0.001) 
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Consumer Loop Error: {e}")
                await asyncio.sleep(1) 
                
    finally:
        await redis_conn.close()

# --- Endpoints ---
@app.get("/status")
async def get_status():
    redis_status = "unknown"
    try:
        if await app.state.redis.ping():
            redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "online",
        "service": "Guardian AI Engine API",
        "redis": redis_status,
        "database": "postgres"
    }

@app.get("/api/alerts")
async def get_alerts(limit: int = 100, db: AsyncSession = Depends(get_db)):
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

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Dashboard'dan gelen ping/mesajları karşıla (bağlantıyı açık tutar)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)