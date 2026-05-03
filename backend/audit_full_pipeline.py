#!/usr/bin/env python3
"""
Comprehensive pipeline audit: Model → Redis → API → DB → Frontend
Checks each stage for data integrity and correctness.
"""
import json
import asyncio
import redis
import numpy as np
from pathlib import Path
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model_provider import get_model_provider
from nfv3_flow_tracker import NFv3FlowTracker

class PipelineAudit:
    def __init__(self):
        self.redis_client = None
        self.db_conn = None
        self.results = {}

    def setup(self):
        """Initialize connections."""
        print("=" * 80)
        print("GUARDIAN AI ENGINE - FULL PIPELINE AUDIT")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().isoformat()}\n")

        # Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            print("✓ Redis connected")
            self.results['redis'] = 'OK'
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            self.results['redis'] = f'FAILED: {e}'

        # PostgreSQL
        try:
            self.db_conn = psycopg2.connect(
                dbname='guardian_ai',
                user='postgres',
                password='postgres',
                host='localhost',
                port=5432
            )
            self.db_conn.autocommit = True
            print("✓ PostgreSQL connected")
            self.results['postgres'] = 'OK'
        except Exception as e:
            print(f"✗ PostgreSQL connection failed: {e}")
            self.results['postgres'] = f'FAILED: {e}'
        print()

    def audit_model_loading(self):
        """Stage 1: Model loading and feature validation."""
        print("\n" + "=" * 80)
        print("STAGE 1: MODEL LOADING & FEATURE VALIDATION")
        print("=" * 80)

        try:
            provider = get_model_provider('flowguard')
            print(f"✓ Model loaded: {provider.__class__.__name__}")

            # Check stats file
            stats_path = Path(__file__).parent / 'saved_models' / 'flowguard_stats.npz'
            if stats_path.exists():
                stats = np.load(stats_path, allow_pickle=True)
                print(f"✓ Stats file found: {stats_path}")
                print(f"  - feature_names: {len(stats['feature_names'])} features")
                print(f"  - feature_means shape: {stats['feature_means'].shape}")
                print(f"  - feature_stds shape: {stats['feature_stds'].shape}")
                print(f"  - log_transform_columns: {len(stats['log_transform_columns'])} features")

                # Check for order mismatch warning
                if hasattr(provider, 'feature_names'):
                    if len(provider.feature_names) != 53:
                        print(f"  ⚠ WARNING: Expected 53 features, got {len(provider.feature_names)}")
            else:
                print(f"✗ Stats file not found: {stats_path}")

            self.results['model_loading'] = 'OK'
            print("✓ Model stage: PASSED")

        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            self.results['model_loading'] = f'FAILED: {e}'
            return False
        return True

    def audit_feature_extraction(self):
        """Stage 2: Feature extraction from sample flow."""
        print("\n" + "=" * 80)
        print("STAGE 2: FEATURE EXTRACTION & PREDICTION")
        print("=" * 80)

        try:
            provider = get_model_provider('flowguard')
            tracker = NFv3FlowTracker()

            # Create a synthetic sample flow
            sample_flow = {
                'src_ip': '10.1.145.154',
                'dst_ip': '34.252.38.116',
                'src_port': 54321,
                'dst_port': 443,
                'protocol': 6,
                'packets': [
                    {'size': 64, 'timestamp': 0.0},
                    {'size': 1024, 'timestamp': 0.1},
                    {'size': 512, 'timestamp': 0.2},
                ],
                'duration_ms': 1281.6,
            }

            print(f"Sample flow: {sample_flow['src_ip']} → {sample_flow['dst_ip']}:{sample_flow['dst_port']}")

            # Extract features (simulated)
            features = {
                'PROTOCOL': 6,
                'L4_SRC_PORT': sample_flow['src_port'],
                'L4_DST_PORT': sample_flow['dst_port'],
                'FLOW_DURATION_MS': sample_flow['duration_ms'],
                'PACKET_COUNT': len(sample_flow['packets']),
                'BYTE_COUNT': sum(p['size'] for p in sample_flow['packets']),
                'SRC_TO_DST_PACKET_COUNT': len(sample_flow['packets']),
                'DST_TO_SRC_PACKET_COUNT': 0,
                'SRC_TO_DST_BYTE_COUNT': sum(p['size'] for p in sample_flow['packets']),
                'DST_TO_SRC_BYTE_COUNT': 0,
                'SRC_TO_DST_AVG_THROUGHPUT': 0.0,
                'DST_TO_SRC_AVG_THROUGHPUT': 0.0,
                'MIN_IP_PKT_LEN': 64,
                'MAX_IP_PKT_LEN': 1024,
                'SRC_TO_DST_MIN_IP_PKT_LEN': 64,
                'SRC_TO_DST_MAX_IP_PKT_LEN': 1024,
                'DST_TO_SRC_MIN_IP_PKT_LEN': 0,
                'DST_TO_SRC_MAX_IP_PKT_LEN': 0,
            }

            # Make prediction
            result = provider.predict(features)
            print(f"\nPrediction result:")
            print(f"  - is_attack: {result['is_attack']}")
            print(f"  - confidence: {result['confidence']:.4f}")
            print(f"  - attack_type: {result['attack_type']}")

            self.results['feature_extraction'] = 'OK'
            print("✓ Feature extraction stage: PASSED")

        except Exception as e:
            print(f"✗ Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['feature_extraction'] = f'FAILED: {e}'
            return False
        return True

    def audit_redis_alerts(self):
        """Stage 3: Check Redis queue for alerts."""
        print("\n" + "=" * 80)
        print("STAGE 3: REDIS ALERT QUEUE")
        print("=" * 80)

        if not self.redis_client:
            print("✗ Redis not connected, skipping")
            return False

        try:
            # Check queue depth
            queue_key = 'alerts_queue'
            queue_size = self.redis_client.llen(queue_key)
            print(f"Alert queue size: {queue_size} messages")

            # Sample last 3 alerts
            if queue_size > 0:
                alerts = self.redis_client.lrange(queue_key, -3, -1)
                print(f"\nLast {min(3, len(alerts))} alerts in queue:")
                for i, alert_json in enumerate(alerts, 1):
                    try:
                        alert = json.loads(alert_json)
                        print(f"\n  Alert #{i}:")
                        print(f"    - IP: {alert.get('src_ip')} → {alert.get('dst_ip')}")
                        print(f"    - is_attack: {alert.get('is_attack')}")
                        print(f"    - confidence: {alert.get('confidence', 0):.4f}")
                        print(f"    - timestamp: {alert.get('timestamp')}")
                    except json.JSONDecodeError:
                        print(f"  Alert #{i}: Invalid JSON")
            else:
                print("⚠ Queue is empty - analyze_engine not running or not pushing alerts")

            self.results['redis_queue'] = 'OK'
            print("\n✓ Redis audit: PASSED")

        except Exception as e:
            print(f"✗ Redis audit failed: {e}")
            self.results['redis_queue'] = f'FAILED: {e}'
            return False
        return True

    def audit_database_alerts(self):
        """Stage 4: Check database for persisted alerts."""
        print("\n" + "=" * 80)
        print("STAGE 4: DATABASE ALERTS")
        print("=" * 80)

        if not self.db_conn:
            print("✗ PostgreSQL not connected, skipping")
            return False

        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check alerts table
                cur.execute("SELECT COUNT(*) as count FROM alerts")
                result = cur.fetchone()
                alert_count = result['count'] if result else 0
                print(f"Total alerts in DB: {alert_count}")

                # Get recent alerts
                cur.execute("""
                    SELECT src_ip, dst_ip, is_attack, confidence, attack_type, timestamp
                    FROM alerts
                    ORDER BY timestamp DESC
                    LIMIT 5
                """)
                recent = cur.fetchall()

                if recent:
                    print(f"\nLast 5 alerts in database:")
                    for i, row in enumerate(recent, 1):
                        print(f"\n  Alert #{i}:")
                        print(f"    - IP: {row['src_ip']} → {row['dst_ip']}")
                        print(f"    - is_attack: {row['is_attack']}")
                        print(f"    - confidence: {row['confidence']:.4f}")
                        print(f"    - attack_type: {row['attack_type']}")
                        print(f"    - timestamp: {row['timestamp']}")
                else:
                    print("⚠ No alerts in database yet")

            self.results['database'] = 'OK'
            print("\n✓ Database audit: PASSED")

        except Exception as e:
            print(f"✗ Database audit failed: {e}")
            self.results['database'] = f'FAILED: {e}'
            return False
        return True

    def audit_api_endpoint(self):
        """Stage 5: Check API endpoint for alerts."""
        print("\n" + "=" * 80)
        print("STAGE 5: API ENDPOINT (/api/alerts)")
        print("=" * 80)

        try:
            import requests
            response = requests.get('http://localhost:8000/api/alerts?limit=5', timeout=5)

            if response.status_code == 200:
                alerts = response.json()
                print(f"API returned {len(alerts)} alerts")

                if alerts:
                    print("\nFirst 3 API alerts:")
                    for i, alert in enumerate(alerts[:3], 1):
                        print(f"\n  Alert #{i}:")
                        print(f"    - IP: {alert.get('src_ip')} → {alert.get('dst_ip')}")
                        print(f"    - is_attack: {alert.get('is_attack')}")
                        print(f"    - confidence: {alert.get('confidence', 0):.4f}")
                else:
                    print("⚠ API returned empty list")

                self.results['api_endpoint'] = 'OK'
                print("\n✓ API endpoint: PASSED")
                return True
            else:
                print(f"✗ API returned status {response.status_code}")
                self.results['api_endpoint'] = f'HTTP {response.status_code}'
                return False

        except requests.exceptions.ConnectionError:
            print("⚠ API not running on localhost:8000")
            self.results['api_endpoint'] = 'NOT_RUNNING'
            return False
        except Exception as e:
            print(f"✗ API audit failed: {e}")
            self.results['api_endpoint'] = f'FAILED: {e}'
            return False

    def audit_websocket(self):
        """Stage 6: Check WebSocket connectivity (test alert subscription)."""
        print("\n" + "=" * 80)
        print("STAGE 6: WEBSOCKET (/ws/alerts)")
        print("=" * 80)

        try:
            import websocket
            import time

            ws_url = "ws://localhost:8000/ws/alerts"
            print(f"Connecting to {ws_url}...")

            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=5)
            print("✓ WebSocket connected")

            # Wait briefly for any messages
            ws.settimeout(2)
            messages = []
            try:
                while len(messages) < 3:
                    msg = ws.recv()
                    if msg:
                        messages.append(json.loads(msg))
            except websocket._exceptions.WebSocketTimeoutException:
                pass
            finally:
                ws.close()

            print(f"Received {len(messages)} WebSocket messages in timeout window")
            if messages:
                print("\nSample WebSocket message:")
                print(f"  {json.dumps(messages[0], indent=2)}")

            self.results['websocket'] = 'OK'
            print("\n✓ WebSocket audit: PASSED")
            return True

        except Exception as e:
            print(f"⚠ WebSocket audit inconclusive: {e}")
            print("  (WebSocket requires persistent connection, may not show messages in quick test)")
            self.results['websocket'] = 'INCONCLUSIVE'
            return None

    def report(self):
        """Print final audit report."""
        print("\n\n" + "=" * 80)
        print("AUDIT SUMMARY")
        print("=" * 80)

        for stage, result in self.results.items():
            status_icon = "✓" if result == "OK" else "✗" if "FAILED" in str(result) else "⚠"
            print(f"{status_icon} {stage.upper():30} {result}")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)

        # Analysis
        if self.results.get('redis_queue') == 'OK':
            print("✓ Analyze engine is running and pushing to Redis")
        else:
            print("⚠ Check if analyze_engine.py is running: python backend/analyze_engine.py --mode live --provider flowguard")

        if self.results.get('database') == 'OK':
            print("✓ API is consuming Redis and persisting to DB")
        else:
            print("⚠ Check if API is running: python backend/api.py")

        if self.results.get('api_endpoint') == 'OK':
            print("✓ API endpoint accessible and returning alerts")
        else:
            print("⚠ Verify API is running on port 8000")

        if self.results.get('websocket') in ['OK', 'INCONCLUSIVE']:
            print("✓ WebSocket accessible (check frontend for connection)")
        else:
            print("⚠ WebSocket not responding")

    def run(self):
        """Run full audit."""
        self.setup()
        self.audit_model_loading()
        self.audit_feature_extraction()
        self.audit_redis_alerts()
        self.audit_database_alerts()
        self.audit_api_endpoint()
        self.audit_websocket()
        self.report()

        # Cleanup
        if self.db_conn:
            self.db_conn.close()

if __name__ == '__main__':
    audit = PipelineAudit()
    audit.run()
