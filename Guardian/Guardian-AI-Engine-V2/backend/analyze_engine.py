import os
import json
import time
import pandas as pd
import numpy as np
import redis
import joblib
import argparse
import asyncio
from feature_extractor import FeatureExtractor, MAPPING

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALERT_QUEUE = "alerts_queue" # Redis List for Persistent Queue
THRESHOLD = 0.40 # Increased from 0.50 to reduce false positives from aggressive adaptation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "base_rf_2017.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler_base.pkl")
# Default Dataset for Simulation
# ESKİ HALİNİ YORUMA AL VEYA SİL, BUNU YAZ:
DEFAULT_DATASET = os.path.join(BASE_DIR, "Wednesday-workingHours.pcap_ISCX.csv")
class TrafficEngine:
    def __init__(self, mode="simulation", file_path=None):
        self.mode = mode
        self.file_path = file_path
        self.redis_client = None
        self.model = None
        self.extractor = None

    def initialize(self):
        """Connect to Redis and Load Models"""
        # 1. Redis
        print(f"[Engine] Connecting to Redis at {REDIS_URL}...")
        self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            self.redis_client.ping()
            print("[Engine] Connected to Redis.")
        except redis.ConnectionError:
            print("[Engine] Error: Could not connect to Redis. Exiting.")
            exit(1)

        # 2. Models
        print(f"[Engine] Loading Models from {MODEL_PATH}...")
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            self.model = joblib.load(MODEL_PATH)
            
            # --- ONLINE ADAPTATION LOGIC (REDEFINED) ---
            ADAPT_FILE = self.file_path
            
            if ADAPT_FILE and os.path.exists(ADAPT_FILE):
                print(f"\n[Engine] ---------------------------------------------------------------")
                print(f"[Engine] ADAPTATION START: Retraining on 5% of {os.path.basename(ADAPT_FILE)}")
                
                try:
                    import numpy as np
                    
                    # 1. Load Calibration Data (Every 20th row)
                    calib_skip_lambda = lambda x: x > 0 and x % 20 != 0
                    
                    df_calib = pd.read_csv(ADAPT_FILE, skiprows=calib_skip_lambda)
                    df_calib.columns = df_calib.columns.str.strip()
                    df_calib = df_calib.rename(columns=MAPPING)
                    
                    if 'Destination Port' in df_calib.columns:
                        df_calib = df_calib[pd.to_numeric(df_calib['Destination Port'], errors='coerce').notnull()]

                    if 'Label' not in df_calib.columns:
                        raise ValueError("Label column missing in calibration data")

                    y_calib = np.where(df_calib['Label'].astype(str).str.lower() == 'benign', 0, 1)
                    
                    X_calib = df_calib.drop(columns=['Label'])
                    valid_cols = [c for c in MAPPING.values() if c in X_calib.columns]
                    X_calib = X_calib[valid_cols]
                    
                    for col in X_calib.columns:
                        X_calib[col] = pd.to_numeric(X_calib[col], errors='coerce').astype('float32')
                    X_calib.replace([np.inf, -np.inf], 0, inplace=True)
                    X_calib.fillna(0, inplace=True)

                    X_calib_scaled = scaler.transform(X_calib) 
                    
                    unique_classes = np.unique(y_calib)
                    n_attack = np.sum(y_calib == 1)
                    print(f"[Engine] Calibration Sample: {len(y_calib)} rows (Attacks Found: {n_attack})")

                    if len(unique_classes) < 2:
                        print(f"[Engine] SKIP RETRAINING: Data contains only ONE class ({unique_classes}).")
                    else:
                        current_trees = self.model.n_estimators
                        new_trees = current_trees + 50
                        
                        self.model.n_estimators = new_trees
                        self.model.fit(X_calib_scaled, y_calib)
                        print(f"[Engine] SUCCESS: Model updated. Forest size expanded: {current_trees} -> {new_trees} trees.")
                        
                except Exception as e:
                    print(f"[Engine] ADAPTATION ERROR: {e}")
                print(f"[Engine] ---------------------------------------------------------------\n")
            else:
                print("[Engine] No adaptation dataset found. Using Base Model 2017.")

            self.extractor = FeatureExtractor(scaler)
            print("[Engine] Models Ready.")

    def process_packet(self, features: dict, source_meta: str, ground_truth_label=None):
        """
        Core Logic: Raw Features -> Feature Extractor -> Model -> Redis Alert
        """
        try:
            # 1. Analysis
            X_input = self.extractor.transform(features)
            probs = self.model.predict_proba(X_input)[0]
            p_attack = float(probs[1]) 
            
            is_attack = p_attack > THRESHOLD
            
            # 2. Construct Result
            result = {
                "timestamp": time.time(),
                "source": source_meta,
                "is_attack": is_attack,
                "confidence": p_attack,
                "label": "Malicious" if is_attack else "Benign", 
                "src_ip": source_meta, # Frontend uyumluluğu için eklendi
                "original_features": features
            }

            # 3. Publish to Redis
            self.redis_client.rpush(ALERT_QUEUE, json.dumps(result))
            self.redis_client.publish("alerts", json.dumps(result)) # PubSub için de atalım
            
            # Log with Ground Truth for Debugging
            if ground_truth_label:
                is_actually_attack = "benign" not in str(ground_truth_label).lower()
                
                if is_actually_attack and not is_attack:
                    print(f"\033[93m[MISS] IP: {source_meta} | Truth: {ground_truth_label} | Pred: Benign | Conf: {p_attack:.4f}\033[0m")
                elif not is_actually_attack and is_attack:
                    print(f"\033[91m[FALSE ALARM] IP: {source_meta} | Truth: Benign | Pred: Attack | Conf: {p_attack:.4f}\033[0m")
                elif is_actually_attack and is_attack:
                    print(f"\033[92m[HIT] IP: {source_meta} | Truth: {ground_truth_label} | Pred: Attack | Conf: {p_attack:.4f}\033[0m")
                else:
                    if np.random.rand() < 0.10:
                         print(f"\033[90m[OK] IP: {source_meta} | Traffic Normal | Conf: {1-p_attack:.4f}\033[0m")

        except Exception as e:
            print(f"Error processing packet: {e}")

    def run_simulation(self):
        """Read from CSV and simulate traffic (Infinite Loop)"""
        if not os.path.exists(self.file_path):
            print(f"Error: Dataset not found at {self.file_path}")
            return

        print(f"[Engine] Starting Simulation with {os.path.basename(self.file_path)}...")
        
        # Daha akıcı bir akış için chunk size'ı düşürdük
        chunk_size = 10 
        total_processed = 0
        
        # --- SONSUZ DÖNGÜ ---
        while True:
            try:
                for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
                    chunk.columns = chunk.columns.str.strip()
                    
                    labels = chunk['Label'].values if 'Label' in chunk.columns else [None]*len(chunk)
                    
                    source_ips = ["Unknown"] * len(chunk)
                    if 'Src IP' in chunk.columns:
                        source_ips = chunk['Src IP'].values
                    elif 'Source IP' in chunk.columns:
                        source_ips = chunk['Source IP'].values
                    
                    chunk = chunk.rename(columns=MAPPING)
                    
                    records = chunk.to_dict(orient='records')
                    
                    for i, record in enumerate(records):
                        src_ip = source_ips[i] if i < len(source_ips) else "192.168.1.10" # IP yoksa uydur
                        self.process_packet(record, f"{src_ip}", ground_truth_label=labels[i])
                    
                    total_processed += len(records)
                    print(f"[Engine] Processed: {total_processed} packets...", end='\r')
                    
                    # Veri akış hızı (Dashboard'a yetişmesi için 0.5sn ideal)
                    time.sleep(0.5)
                
                print("\n[Engine] Dataset finished. Restarting simulation loop...")
                time.sleep(1) # Başa sarmadan önce az bekle

            except Exception as e:
                print(f"[Engine] Error in simulation loop: {e}")
                time.sleep(2)

    def run_live(self):
        """
        Placeholder for Scapy Sniffing.
        """
        print("[Engine] Starting Live Capture (SCAPY Placeholder)...")
        while True:
            time.sleep(1)

    def start(self):
        self.initialize()
        if self.mode == 'simulation':
            self.run_simulation()
        else:
            self.run_live()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['simulation', 'live'], default='simulation', help="Operation mode")
    parser.add_argument("--file", default=DEFAULT_DATASET, help="Path to CSV for simulation")
    args = parser.parse_args()
    
    engine = TrafficEngine(mode=args.mode, file_path=args.file)
    engine.start()