import subprocess
import time
import redis
import requests
import json
import sys
import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Adjust connection string if needed
DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/guardian_ai"

def run_verification():
    print("Starting DB verification...")
    
    # 1. Start API Server
    print("Starting API server...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--port", "8002"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for server to start
        time.sleep(5)
        
        # 2. Connect to Redis (to push alert)
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        try:
            r.ping()
        except redis.ConnectionError:
            print("Error: Redis is not running. Please start Redis.")
            return

        r.delete("alerts_queue")
        
        # 3. Simulate an Alert
        mock_alert = {
            "timestamp": time.time(),
            "source": "10.0.0.5",
            "is_attack": True,
            "confidence": 0.99,
            "attack_type": "SQL Injection",
            "original_features": {"payload": "' OR 1=1 --"}
        }
        
        print("Pushing mock alert to Redis...")
        r.rpush("alerts_queue", json.dumps(mock_alert))
        
        # Wait for consumer to process and save to DB
        time.sleep(3)
        
        # 4. Verify /api/alerts (which queries DB)
        print("Verifying /api/alerts...")
        try:
            resp = requests.get("http://localhost:8002/api/alerts")
            if resp.status_code == 200:
                alerts = resp.json()
                # find our alert
                found = False
                for a in alerts:
                    if a.get("attack_type") == "SQL Injection" and a.get("source") == "10.0.0.5":
                        found = True
                        break
                
                if found:
                    print("✅ /api/alerts returned correctly from DB.")
                else:
                    print(f"❌ Alert not found in DB response. Got: {alerts}")
            else:
                print(f"❌ /api/alerts returned status {resp.status_code}")
                # Print server stderr for debugging
                # (Reading stderr non-blocking is tricky without threads, but for verification script 
                # we can just assume if it failed we might want to check logs manually if this wasn't enough)
        except Exception as e:
            print(f"❌ /api/alerts request failed: {e}")

    finally:
        # Cleanup
        print("Stopping API server...")
        if os.name == 'nt':
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(api_process.pid)])
        else:
            api_process.terminate()

if __name__ == "__main__":
    run_verification()
