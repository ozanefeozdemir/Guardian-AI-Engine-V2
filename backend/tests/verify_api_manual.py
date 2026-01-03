import asyncio
from api import app, lifespan, predict, PredictionRequest

async def main():
    print("--- Starting Manual API Verification ---")
    
    # 1. Start Lifespan (Load Models)
    async with lifespan(app):
        print("[Check] Lifespan started. Checking model status...")
        if app.state.model:
            print("    -> Model loaded successfully.")
        else:
            print("    -> [ERROR] Model NOT loaded.")
            return

        # 2. Test Prediction
        print("\n[Check] Testing /predict endpoint...")
        payload = {
            "Destination Port": 80,
            "Total Fwd Packets": 10,
            "Flow Bytes/s": 500.0
        }
        
        req = PredictionRequest(features=payload)
        
        try:
            response = await predict(req)
            print(f"    -> Prediction Result: {response}")
            print("    -> SUCCESS: Prediction complete.")
        except Exception as e:
            print(f"    -> [ERROR] Prediction failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
