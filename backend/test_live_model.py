"""
Test Live Model - FlowGuard Provider Standalone Test
=====================================================
Captures live packets, extracts NF-v3 flow features, runs them through
the FlowGuard model, and prints each flow with its prediction.

No Redis or Docker required - runs entirely standalone.

Usage:
    python test_live_model.py                    # sniff 300 packets
    python test_live_model.py --count 500        # sniff 500 packets
    python test_live_model.py --timeout 60       # sniff for 60 seconds
    python test_live_model.py --provider legacy  # test legacy provider
"""

import os
import sys
import io
import time
import argparse
import numpy as np

# Fix Windows encoding for file redirect
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add backend to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scapy.all import sniff, IP, TCP, UDP
from nfv3_flow_tracker import NFv3FlowTracker
from packet_flow import CICFlowTracker
from model_provider import get_model_provider


class LiveModelTester:
    """
    Standalone tester: captures packets → builds flows → runs model → prints results.
    No Redis needed.
    """

    def __init__(self, provider_name='flowguard', count=300, timeout_sec=None):
        self.provider_name = provider_name
        self.count = count
        self.timeout_sec = timeout_sec
        self.provider = None

        # Counters
        self.total_flows = 0
        self.benign_flows = 0
        self.attack_flows = 0
        self.flow_results = []

    def initialize(self):
        """Load model provider (no Redis)."""
        print("=" * 80)
        print(f"  FlowGuard Live Model Tester")
        print(f"  Provider: {self.provider_name}")
        print(f"  Packet count: {self.count or 'unlimited'}")
        print(f"  Time limit: {self.timeout_sec or 'none'}s")
        print("=" * 80)
        print()

        print(f"[Tester] Loading model provider '{self.provider_name}'...")
        self.provider = get_model_provider(self.provider_name)
        self.provider.load()
        info = self.provider.get_info()
        print(f"[Tester] Provider ready: {info}")
        print()

    def on_flow_ready(self, features, src_ip, dst_ip):
        """Called when a flow closes. Runs prediction and prints results."""
        self.total_flows += 1
        flow_num = self.total_flows

        try:
            result = self.provider.predict(features)
            is_attack = result['is_attack']
            confidence = result['confidence']
            attack_type = result.get('attack_type', 'Unknown')

            if is_attack:
                self.attack_flows += 1
                label_color = "\033[91m"  # Red
                label_str = "[!] ATTACK"
            else:
                self.benign_flows += 1
                label_color = "\033[92m"  # Green
                label_str = "[OK] BENIGN"

            reset = "\033[0m"
            dim = "\033[90m"

            # Print flow summary
            print(f"\n{'-' * 70}")
            print(f"  Flow #{flow_num}: {src_ip} -> {dst_ip}")
            print(f"  {label_color}Label: {label_str} ({attack_type}){reset}  |  "
                  f"Confidence: {confidence:.4f}")

            # Print key features for diagnosis
            if self.provider_name == 'flowguard':
                self._print_nfv3_features(features, dim, reset)
            else:
                self._print_cic_features(features, dim, reset)

            # Store result
            self.flow_results.append({
                'flow_num': flow_num,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'is_attack': is_attack,
                'confidence': confidence,
                'attack_type': attack_type,
                'key_features': self._extract_key_features(features),
            })

        except Exception as e:
            print(f"\n[ERROR] Flow #{flow_num} {src_ip}->{dst_ip}: {e}")

    def _print_nfv3_features(self, features, dim, reset):
        """Print key NF-v3 features for diagnosis."""
        print(f"  {dim}Protocol: {features.get('PROTOCOL', '?')}  |  "
              f"L7: {features.get('L7_PROTO', '?')}  |  "
              f"Duration: {features.get('FLOW_DURATION_MILLISECONDS', 0):.1f}ms{reset}")
        print(f"  {dim}IN: {features.get('IN_BYTES', 0)}B/{features.get('IN_PKTS', 0)}pkts  |  "
              f"OUT: {features.get('OUT_BYTES', 0)}B/{features.get('OUT_PKTS', 0)}pkts{reset}")
        print(f"  {dim}Ports: {features.get('L4_SRC_PORT', '?')} -> {features.get('L4_DST_PORT', '?')}  |  "
              f"TCP Flags: {features.get('TCP_FLAGS', 0)}  |  "
              f"TTL: {features.get('MIN_TTL', 0)}-{features.get('MAX_TTL', 0)}{reset}")
        print(f"  {dim}Throughput: S->D={features.get('SRC_TO_DST_AVG_THROUGHPUT', 0):.0f} bps  |  "
              f"D->S={features.get('DST_TO_SRC_AVG_THROUGHPUT', 0):.0f} bps{reset}")
        print(f"  {dim}Retransmit: IN={features.get('RETRANSMITTED_IN_PKTS', 0)} pkts  |  "
              f"OUT={features.get('RETRANSMITTED_OUT_PKTS', 0)} pkts{reset}")

    def _print_cic_features(self, features, dim, reset):
        """Print key CIC-IDS features for diagnosis."""
        print(f"  {dim}Protocol: {features.get('Protocol', '?')}  |  "
              f"Duration: {features.get('Flow Duration', 0):.0f}us{reset}")
        print(f"  {dim}Fwd: {features.get('Tot Fwd Pkts', 0)} pkts  |  "
              f"Bwd: {features.get('Tot Bwd Pkts', 0)} pkts{reset}")
        print(f"  {dim}Port: {features.get('Dst Port', '?')}  |  "
              f"Byts/s: {features.get('Flow Byts/s', 0):.0f}  |  "
              f"Pkts/s: {features.get('Flow Pkts/s', 0):.0f}{reset}")

    def _extract_key_features(self, features):
        """Extract key features for summary table."""
        if self.provider_name == 'flowguard':
            return {
                'protocol': features.get('PROTOCOL', 0),
                'duration_ms': features.get('FLOW_DURATION_MILLISECONDS', 0),
                'in_bytes': features.get('IN_BYTES', 0),
                'out_bytes': features.get('OUT_BYTES', 0),
                'in_pkts': features.get('IN_PKTS', 0),
                'out_pkts': features.get('OUT_PKTS', 0),
                'src_port': features.get('L4_SRC_PORT', 0),
                'dst_port': features.get('L4_DST_PORT', 0),
                'tcp_flags': features.get('TCP_FLAGS', 0),
            }
        else:
            return {
                'protocol': features.get('Protocol', 0),
                'duration': features.get('Flow Duration', 0),
                'fwd_pkts': features.get('Tot Fwd Pkts', 0),
                'bwd_pkts': features.get('Tot Bwd Pkts', 0),
                'dst_port': features.get('Dst Port', 0),
            }

    def run(self):
        """Main entry: initialize model, capture packets, print summary."""
        self.initialize()

        # Select tracker based on provider
        use_nfv3 = (self.provider_name == 'flowguard')
        tracker_name = "NF-v3" if use_nfv3 else "CIC-IDS"

        if use_nfv3:
            tracker = NFv3FlowTracker(timeout=120.0, on_flow_ready=self.on_flow_ready)
        else:
            tracker = CICFlowTracker(timeout=120.0, on_flow_ready=self.on_flow_ready)

        def packet_callback(pkt):
            if IP in pkt:
                tracker.process_packet(pkt)

        # BPF filter to exclude infrastructure traffic
        bpf_filter = "not (port 6379 or port 5432 or port 8000)"

        print(f"[Tester] Starting {tracker_name} flow capture...")
        print(f"[Tester] BPF filter: {bpf_filter}")
        print(f"[Tester] Capturing {self.count or 'unlimited'} packets... (Ctrl+C to stop early)")
        print()

        start_time = time.time()
        try:
            sniff(
                prn=packet_callback,
                store=False,
                filter=bpf_filter,
                count=self.count or 0,
                timeout=self.timeout_sec,
            )
        except KeyboardInterrupt:
            print("\n\n[Tester] Interrupted by user.")

        # Flush remaining flows
        print("\n[Tester] Flushing remaining active flows...")
        tracker.flush_all()

        elapsed = time.time() - start_time
        self._print_summary(elapsed)

    def _print_summary(self, elapsed):
        """Print final summary of test results."""
        print()
        print("=" * 80)
        print("  TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"  Duration:      {elapsed:.1f} seconds")
        print(f"  Total Flows:   {self.total_flows}")
        print()

        if self.total_flows == 0:
            print("  No flows captured. Try increasing --count or --timeout.")
            print("=" * 80)
            return

        benign_pct = (self.benign_flows / self.total_flows) * 100
        attack_pct = (self.attack_flows / self.total_flows) * 100

        print(f"  \033[92m[OK] Benign:  {self.benign_flows:4d}  ({benign_pct:5.1f}%)\033[0m")
        print(f"  \033[91m[!!] Attack:  {self.attack_flows:4d}  ({attack_pct:5.1f}%)\033[0m")
        print()

        # Health check
        if attack_pct > 30:
            print("  \033[91m[!!] WARNING: High attack rate ({:.1f}%)! This is likely a false-positive issue.\033[0m".format(attack_pct))
            print("  \033[91m  Normal benign traffic should be >90% benign.\033[0m")
        elif attack_pct > 10:
            print("  \033[93m[!] NOTICE: Elevated attack rate ({:.1f}%). Review flagged flows.\033[0m".format(attack_pct))
        else:
            print("  \033[92m[OK] Attack rate looks healthy ({:.1f}%).\033[0m".format(attack_pct))

        print()

        # Show confidence distribution for attack flows
        if self.flow_results:
            attack_confs = [r['confidence'] for r in self.flow_results if r['is_attack']]
            benign_confs = [1 - r['confidence'] for r in self.flow_results if not r['is_attack']]

            if attack_confs:
                print(f"  Attack confidence distribution:")
                print(f"    Min: {min(attack_confs):.4f}  Max: {max(attack_confs):.4f}  "
                      f"Mean: {np.mean(attack_confs):.4f}  Std: {np.std(attack_confs):.4f}")
            if benign_confs:
                print(f"  Benign confidence distribution:")
                print(f"    Min: {min(benign_confs):.4f}  Max: {max(benign_confs):.4f}  "
                      f"Mean: {np.mean(benign_confs):.4f}  Std: {np.std(benign_confs):.4f}")

        print()

        # Detailed table of all flows
        print("  DETAILED FLOW TABLE:")
        print(f"  {'#':>4}  {'Src IP':<16}  {'Dst IP':<16}  {'Label':<10}  {'Conf':>6}  {'Details'}")
        print(f"  {'-' * 4}  {'-' * 16}  {'-' * 16}  {'-' * 10}  {'-' * 6}  {'-' * 30}")

        for r in self.flow_results:
            label = "ATTACK" if r['is_attack'] else "Benign"
            color = "\033[91m" if r['is_attack'] else "\033[92m"
            reset = "\033[0m"
            kf = r['key_features']

            if self.provider_name == 'flowguard':
                detail = (f"proto={kf.get('protocol', '?')} "
                          f"dur={kf.get('duration_ms', 0):.0f}ms "
                          f"port={kf.get('src_port', '?')}->{kf.get('dst_port', '?')} "
                          f"in={kf.get('in_bytes', 0)}B out={kf.get('out_bytes', 0)}B")
            else:
                detail = (f"proto={kf.get('protocol', '?')} "
                          f"dur={kf.get('duration', 0):.0f}us "
                          f"fwd={kf.get('fwd_pkts', 0)} bwd={kf.get('bwd_pkts', 0)}")

            print(f"  {color}{r['flow_num']:>4}  {r['src_ip']:<16}  {r['dst_ip']:<16}  "
                  f"{label:<10}  {r['confidence']:.4f}  {detail}{reset}")

        print()
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FlowGuard model on live traffic (standalone, no Redis)")
    parser.add_argument("--provider", choices=['flowguard', 'legacy', 'guardian', 'placeholder'],
                        default='flowguard', help="Model provider to test (default: flowguard)")
    parser.add_argument("--count", type=int, default=300,
                        help="Number of packets to capture (default: 300)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max seconds to capture (default: no limit)")
    args = parser.parse_args()

    tester = LiveModelTester(
        provider_name=args.provider,
        count=args.count,
        timeout_sec=args.timeout,
    )
    tester.run()
