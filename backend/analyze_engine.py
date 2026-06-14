try:    
    import os
    import json
    import time
    import signal
    import pandas as pd
    import numpy as np
    import redis
    import argparse
    import asyncio
    import math
    from scapy.all import sniff
    from scapy.layers.inet import IP, TCP, UDP
    from model_provider import get_model_provider
    from packet_flow import CICFlowTracker
    from nfv3_flow_tracker import NFv3FlowExporter
    from ndpi_resolver import NDPIResolver
    from ip_matcher import IPRuleMatcher
except ImportError:
    print("Error: Missing required libraries. Please install them using 'pip install -r requirements.txt'")
    exit(1)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ALERT_QUEUE = "alerts_queue"
# Global threshold for sigmoid-based models. FlowGuardProvider uses softmax and overrides with its own DECISION_THRESHOLD = 0.50.
THRESHOLD = 0.40 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DEFAULT_DATASET = os.path.join(
    os.path.dirname(BASE_DIR), "model", "canavar-model", "data", "raw", "NF-UNSW-NB15-v3.csv"
)

class TrafficEngine:
    def __init__(self, mode="simulation", file_path=None, provider_name=None, delay=0.05):
        self.mode = mode
        self.file_path = file_path
        self.redis_client = None
        self.provider = None
        self.provider_name = provider_name
        self.matcher = None
        self._shutting_down = False
        # Per-flow pause in simulation mode so alerts stream at a realistic pace
        # (instead of dumping the whole dataset instantly).
        self.delay = delay

        # Handle SIGTERM from docker-compose gracefully
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[Engine] Received {sig_name}, shutting down gracefully...")
        self._shutting_down = True
        raise SystemExit(0)
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
        self.provider = get_model_provider(self.provider_name)
        self.provider.load()

        info = self.provider.get_info()
        print(f"[Engine] Provider: {info['provider']} | Ready: {info['ready']}")

        # 3. IP Rule Matcher (whitelist/blacklist short-circuit)
        self.matcher = IPRuleMatcher(self.redis_client)
        self.matcher.refresh()
        print(f"[Engine] IP rule matcher loaded "
              f"(whitelist: {len(self.matcher._whitelist)}, "
              f"blacklist: {len(self.matcher._blacklist)})")

    def process_packet(self, features: dict, source_meta: str, ground_truth_label=None):
        """
        Core Logic: Raw Features -> Model Provider -> Redis Alert
        """
        try:
            # 0. Clean NaN/Inf from features BEFORE model prediction (defense-in-depth)
            clean_features = {
                k: (0.0 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                for k, v in features.items()
            }

            # 1. IP rule short-circuit (skips ML inference for known-good/bad sources)
            if self.matcher is not None:
                self.matcher.maybe_refresh()
                src_ip = source_meta.split("->")[0].strip()
                verdict, rule_id = self.matcher.match(src_ip)

                if verdict == "whitelist":
                    alert = {
                        "timestamp": time.time(),
                        "source": source_meta,
                        "is_attack": False,
                        "confidence": 0.0,
                        "attack_type": "Whitelisted",
                        "rule_id": rule_id,
                        "original_features": clean_features,
                    }
                    self.redis_client.rpush(ALERT_QUEUE, json.dumps(alert))
                    print(f"\033[96m [WL] IP: {source_meta} | Rule: {rule_id} -> whitelisted\033[0m")
                    return

                if verdict == "blacklist":
                    alert = {
                        "timestamp": time.time(),
                        "source": source_meta,
                        "is_attack": True,
                        "confidence": 1.0,
                        "attack_type": "Blacklisted",
                        "rule_id": rule_id,
                        "original_features": clean_features,
                    }
                    self.redis_client.rpush(ALERT_QUEUE, json.dumps(alert))
                    print(f"\033[95m [BL] IP: {source_meta} | Rule: {rule_id} -> blacklisted\033[0m")
                    return

            # 2. Predict via provider
            result = self.provider.predict(clean_features)

            is_attack = result["is_attack"]
            p_attack = result["confidence"]

            # 3. Construct Result

            alert = {
                "timestamp": time.time(),
                "source": source_meta,
                "is_attack": is_attack,
                "confidence": p_attack,
                "attack_type": result.get("attack_type", "Malicious" if is_attack else "Benign"),
                "original_features": clean_features
            }

            self.redis_client.rpush(ALERT_QUEUE, json.dumps(alert))
            
            # For logging
            if ground_truth_label is not None:
                is_actually_attack = "benign" not in str(ground_truth_label).lower()
                truth_str = str(ground_truth_label)
                pred_str = "Attack" if is_attack else "Benign"
                
                if is_actually_attack and not is_attack:
                    print(f"\033[93m[MISS] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {p_attack:.4f}\033[0m")
                elif not is_actually_attack and is_attack:
                    print(f"\033[91m[FALSE ALARM] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {p_attack:.4f}\033[0m")
                elif is_actually_attack and is_attack:
                    print(f"\033[92m[HIT] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {p_attack:.4f}\033[0m")
                else:
                    print(f"\033[90m[OK] IP: {source_meta} | Truth: {truth_str} | Pred: {pred_str} | Conf: {1-p_attack:.4f}\033[0m")
            else:
                # Live mode logging
                pred_str = "ATTACK" if is_attack else "Benign"
                color = "\033[91m [!] " if is_attack else "\033[92m [OK] "
                
                proto = features.get("PROTOCOL", features.get("Protocol", "?"))
                dur = features.get("FLOW_DURATION_MILLISECONDS", features.get("Flow Duration", 0))
                if isinstance(dur, (int, float)):
                    dur_str = f"{dur:.1f}ms"
                else:
                    dur_str = f"{dur}"
                    
                print(f"{color}IP: {source_meta:<35} | Pred: {pred_str:<8} | Type: {result.get('attack_type', ''):<10} | Conf: {p_attack:.4f} | Proto: {proto} | Dur: {dur_str}\033[0m")

        except redis.ConnectionError:
            if not self._shutting_down:
                print("[Engine] Redis connection lost. Waiting for reconnect...")
                self._shutting_down = True
        except Exception as e:
            if not self._shutting_down:
                print(f"Error processing packet: {e}")

    def run_simulation(self):
        """Replay a raw NF-v3 (NF-UNSW-NB15-v3) CSV through the same pipeline as
        live capture.

        Each CSV row is already a closed-flow record in the exact schema the
        NFv3FlowExporter emits live (IPV4_* identity cols, the 47 numeric flow
        features, FLOW_START/END_MILLISECONDS, plus Label/Attack ground truth).
        We hand those raw records straight to process_packet -> FlowGuardProvider,
        so everything downstream is identical to run_live; only the source of the
        records differs (dataset vs. sniffed packets).
        """
        if not os.path.exists(self.file_path):
            print(f"Error: Dataset not found at {self.file_path}")
            return

        print(f"[Engine] simulating {os.path.basename(self.file_path)} file")

        chunk_size = 500
        total_processed = 0

        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
            chunk.columns = chunk.columns.str.strip()

            # Ground truth: prefer the readable 'Attack' string (Benign / Exploits /
            # DoS / ...), fall back to deriving from numeric 'Label' (0 -> Benign).
            if 'Attack' in chunk.columns:
                attacks = chunk['Attack'].values
            else:
                attacks = [None] * len(chunk)
            labels = chunk['Label'].values if 'Label' in chunk.columns else [None] * len(chunk)

            # NF-v3 identity columns -> source / destination IP for the alert.
            src_ips = chunk['IPV4_SRC_ADDR'].values if 'IPV4_SRC_ADDR' in chunk.columns else ["Unknown"] * len(chunk)
            dst_ips = chunk['IPV4_DST_ADDR'].values if 'IPV4_DST_ADDR' in chunk.columns else ["Unknown"] * len(chunk)

            # Pass raw NF-v3 rows through unchanged (the CIC MAPPING does not apply).
            # Drop the label columns so they are not stored as model features.
            feature_chunk = chunk.drop(columns=[c for c in ('Label', 'Attack') if c in chunk.columns])
            records = feature_chunk.to_dict(orient='records')

            for i, record in enumerate(records):
                src_ip = src_ips[i] if i < len(src_ips) else "Unknown"
                dst_ip = dst_ips[i] if i < len(dst_ips) else "Unknown"

                ground_truth = attacks[i]
                if ground_truth is None or (isinstance(ground_truth, float) and math.isnan(ground_truth)):
                    label = labels[i]
                    if label is not None:
                        ground_truth = "Benign" if str(label).strip() in ("0", "0.0") else "Attack"

                self.process_packet(record, f"{src_ip}->{dst_ip}", ground_truth_label=ground_truth)

                if self.delay > 0:
                    time.sleep(self.delay)

            total_processed += len(records)
            print(f"[Engine] Processed: {total_processed} flows...", end='\r')

    def run_live(self):
        """
        Canli trafik yakalama -- Flow bazli analiz.
        Paketler 5-tuple bazli flow'lara gruplanir, flow kapandiginda
        (FIN/RST veya timeout) feature'lar cikarilip modele gonderilir.

        Provider'a gore farkli tracker kullanilir:
          - flowguard -> NFv3FlowTracker (53 NF-v3 feature)
          - digerleri -> CICFlowTracker (79 CIC-IDS feature)
        """
        # FlowGuard provider icin NF-v3 tracker, digerleri icin CIC tracker
        use_nfv3 = (self.provider_name == 'flowguard')
        tracker_name = "NF-v3" if use_nfv3 else "CIC-IDS"
        print(f"[Engine] Starting Live Capture ({tracker_name} Flow-Based)...")

        def on_flow_ready(features, src_ip, dst_ip):
            """Flow kapandiginda cagirilir -> model prediction -> Redis."""
            self.process_packet(features, f"{src_ip}->{dst_ip}")

        resolver = None
        if use_nfv3:
            resolver = NDPIResolver(interface="en0")
            if not resolver.start():
                resolver = None
            tracker = NFv3FlowExporter(on_flow_ready=on_flow_ready, l7_resolver=resolver)
        else:
            tracker = CICFlowTracker(timeout=120.0, on_flow_ready=on_flow_ready)

        def packet_callback(packet):
            if IP in packet:
                tracker.process_packet(packet)

        print("[Engine] Flow tracking started. (Press Ctrl+C to stop)")
        # Kendi trafigini (Redis, Postgres, API) analiz etmemesi icin BPF filter (Sonsuz dongu ve False Positive engeller)
        bpf_filter = "not (port 6379 or port 5432 or port 8000)"
        
        try:
            sniff(prn=packet_callback, store=False, filter=bpf_filter)
        except KeyboardInterrupt:
            print("\n[Engine] Stopping... Flushing remaining flows.")
            tracker.flush_all()
            if resolver is not None:
                resolver.stop()
            print("[Engine] Done.")
    
    def start(self):
        self.initialize()
        if self.mode == 'simulation':
            self.run_simulation()
        else:
            self.run_live()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['simulation', 'live'], default='simulation', help="Operation mode")
    parser.add_argument("--file", default=DEFAULT_DATASET,
                        help="Path to a raw NF-UNSW-NB15-v3 CSV for simulation "
                             "(default: model/canavar-model/data/raw/NF-UNSW-NB15-v3.csv)")
    parser.add_argument("--provider", choices=['placeholder', 'legacy', 'custom', 'guardian', 'flowguard'], default='flowguard',
                        help="Model provider (default: flowguard — canavar-model). FlowGuard uses argmax; "
                             "no threshold flag because canavar-model has no tunable threshold.")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds to pause between flows in simulation mode "
                             "(default: 0.05). Set to 0 for max speed.")
    args = parser.parse_args()

    # file_path argümanını motorun içine gönderiyoruz:
    engine = TrafficEngine(mode=args.mode, file_path=args.file, provider_name=args.provider, delay=args.delay)
    engine.start()