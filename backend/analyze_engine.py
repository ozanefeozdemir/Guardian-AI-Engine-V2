
import os
import json
import time
import pandas as pd
import numpy as np
import redis
import argparse
import asyncio
from feature_extractor import MAPPING
from model_provider import get_model_provider

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALERT_QUEUE = "alerts_queue" # Redis List for Persistent Queue
THRESHOLD = 0.40 # Increased from 0.50 to reduce false positives from aggressive adaptation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default Dataset for Simulation
DEFAULT_DATASET = os.path.join(BASE_DIR, "datasets", "raw", "CIC-IDS 2017","TrafficLabelling" , "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

class TrafficEngine:
    def __init__(self, mode="simulation", file_path=None, provider_name=None):
        self.mode = mode
        self.file_path = file_path
        print("adapt file: ",self.file_path)
        self.redis_client = None
        self.provider = None
        self.provider_name = provider_name

    def initialize(self):
        """Connect to Redis and Load Model Provider"""
        # 1. Redis
        print(f"[Engine] Connecting to Redis at {REDIS_URL}...")
        self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            self.redis_client.ping()
            print("[Engine] Connected to Redis.")
        except redis.ConnectionError:
            print("[Engine] Error: Could not connect to Redis. Exiting.")
            exit(1)

        # 2. Model Provider
        print(f"[Engine] Initializing Model Provider...")
        provider_kwargs = {}
        if self.provider_name == "legacy":
            provider_kwargs["adapt_file"] = self.file_path

        self.provider = get_model_provider(self.provider_name, **provider_kwargs)
        self.provider.load()

        info = self.provider.get_info()
        print(f"[Engine] Provider: {info['provider']} | Ready: {info['ready']}")

    def process_packet(self, features: dict, source_meta: str, ground_truth_label=None):
        """
        Core Logic: Raw Features -> Model Provider -> Redis Alert
        """
        try:
            # 1. Predict via provider
            result = self.provider.predict(features)

            is_attack = result["is_attack"]
            p_attack = result["confidence"]

            # 2. Construct Result
            # Clean NaN/Inf from features for valid JSON
            import math
            clean_features = {
                k: (0.0 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                for k, v in features.items()
            }
            
            alert = {
                "timestamp": time.time(),
                "source": source_meta,
                "is_attack": is_attack,
                "confidence": p_attack,
                "attack_type": result.get("attack_type", "Malicious" if is_attack else "Benign"), 
                "original_features": clean_features
            }

            # 3. Publish to Redis
            self.redis_client.rpush(ALERT_QUEUE, json.dumps(alert))
            
            # Log with Ground Truth for Debugging
            if ground_truth_label:
                # Determine colors and tags based on correctness
                is_actually_attack = "benign" not in str(ground_truth_label).lower()
                truth_str = str(ground_truth_label)
                pred_str = "Attack" if is_attack else "Benign"
                
                if is_actually_attack and not is_attack:
                    # MISS (Yellow)
                    print(f"\033[93m[MISS] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {p_attack:.4f}\033[0m")
                elif not is_actually_attack and is_attack:
                    # FALSE ALARM (Red)
                    print(f"\033[91m[FALSE ALARM] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {p_attack:.4f}\033[0m")
                elif is_actually_attack and is_attack:
                    # HIT (Green)
                    print(f"\033[92m[HIT] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {p_attack:.4f}\033[0m")
                else:
                    # True Negative (Grey) - Clean Log for all benign traffic
                    print(f"\033[90m[OK] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {1-p_attack:.4f}\033[0m")

        except Exception as e:
            print(f"Error processing packet: {e}")

    def run_simulation(self):
        """Read from CSV and simulate traffic"""
        if not os.path.exists(self.file_path):
            print(f"Error: Dataset not found at {self.file_path}")
            return

        print(f"[Engine] simulating {os.path.basename(self.file_path)} file")
        
        chunk_size = 500
        total_processed = 0
        
        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
            chunk.columns = chunk.columns.str.strip()
            
            # Extract Label before feature mapping (Label might be lost or renamed)
            labels = chunk['Label'].values if 'Label' in chunk.columns else [None]*len(chunk)
            
            # Extract Source IP if available (CIC-IDS 2018 usually has 'Src IP', 2017 has 'Source IP')
            source_ips = ["Unknown"] * len(chunk)
            if 'Src IP' in chunk.columns:
                source_ips = chunk['Src IP'].values
            elif 'Source IP' in chunk.columns:
                source_ips = chunk['Source IP'].values
            
            chunk = chunk.rename(columns=MAPPING)
            
            records = chunk.to_dict(orient='records')
            
            for i, record in enumerate(records):
                src_ip = source_ips[i] if i < len(source_ips) else "Unknown"
                self.process_packet(record, f"{src_ip}", ground_truth_label=labels[i])
            
            total_processed += len(records)
            print(f"[Engine] Processed: {total_processed} packets...", end='\r')

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
    parser.add_argument("--provider", choices=['placeholder', 'legacy', 'custom'], default=None, 
                        help="Model provider (default: reads MODEL_PROVIDER env var, fallback: legacy)")
    args = parser.parse_args()
    
    engine = TrafficEngine(mode=args.mode, file_path=args.file, provider_name=args.provider)
    engine.start()
