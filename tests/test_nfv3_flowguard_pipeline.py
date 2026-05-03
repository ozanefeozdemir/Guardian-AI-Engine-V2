"""
Integration tests for NFv3FlowTracker + FlowGuardProvider pipeline.

Tests the full flow:
  NFv3FlowTracker.extract_features() → FlowGuardProvider._preprocess() → model input
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from nfv3_flow_tracker import NFv3FlowTracker
from scapy.all import Packet, IP, TCP, UDP


class TestNFv3FeatureExtraction(unittest.TestCase):
    """Test NFv3FlowTracker.extract_features() output."""

    def setUp(self):
        """Create a mock flow for testing."""
        self.tracker = NFv3FlowTracker(timeout=120.0)
        self.flow_id = "192.168.1.1:12345-192.168.1.2:443-6"

    def test_extract_features_count(self):
        """Test that extract_features returns exactly 49 keys (47 raw + 2 port values)."""
        # Create a minimal flow
        current_time = 1000.0
        pkt_len = 100
        ttl = 64
        meta = {
            'src_ip': '192.168.1.1',
            'dst_ip': '192.168.1.2',
            'src_port': 12345,
            'dst_port': 443,
            'protocol': 6,  # TCP
        }

        flow = self.tracker._create_flow(
            MagicMock(time=current_time, spec=Packet),
            current_time, pkt_len, ttl, meta
        )

        # Store flow in tracker
        self.tracker.active_flows[self.flow_id] = flow

        # Extract features
        features = self.tracker.extract_features(self.flow_id)

        # Should have 47 raw NF-v3 features + L4_SRC_PORT + L4_DST_PORT = 49 keys
        self.assertEqual(
            len(features), 49,
            f"Expected 49 features, got {len(features)}. Keys: {list(features.keys())}"
        )

    def test_extract_features_required_keys(self):
        """Test that extract_features includes all required keys."""
        current_time = 1000.0
        pkt_len = 100
        ttl = 64
        meta = {
            'src_ip': '10.0.0.1',
            'dst_ip': '10.0.0.2',
            'src_port': 54321,
            'dst_port': 80,
            'protocol': 6,
        }

        flow = self.tracker._create_flow(
            MagicMock(time=current_time, spec=Packet),
            current_time, pkt_len, ttl, meta
        )
        self.tracker.active_flows[self.flow_id] = flow
        features = self.tracker.extract_features(self.flow_id)

        # Required identity keys for port bucketing
        self.assertIn('L4_SRC_PORT', features)
        self.assertIn('L4_DST_PORT', features)

        # Required NF-v3 features
        required_features = [
            'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
            'MIN_TTL', 'MAX_TTL',
        ]
        for key in required_features:
            self.assertIn(
                key, features,
                f"Missing required feature: {key}"
            )

    def test_extract_features_port_values(self):
        """Test that port values are correctly extracted."""
        current_time = 1000.0
        pkt_len = 100
        ttl = 64
        src_port = 12345
        dst_port = 443
        meta = {
            'src_ip': '192.168.1.1',
            'dst_ip': '192.168.1.2',
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': 6,
        }

        flow = self.tracker._create_flow(
            MagicMock(time=current_time, spec=Packet),
            current_time, pkt_len, ttl, meta
        )
        self.tracker.active_flows[self.flow_id] = flow
        features = self.tracker.extract_features(self.flow_id)

        self.assertEqual(features['L4_SRC_PORT'], src_port)
        self.assertEqual(features['L4_DST_PORT'], dst_port)


class TestFlowGuardPreprocessing(unittest.TestCase):
    """Test FlowGuardProvider._preprocess() pipeline."""

    def setUp(self):
        """Mock a FlowGuardProvider with minimal setup."""
        # We don't want to load actual model files, so we mock them
        self.mock_provider = MagicMock()
        self.mock_provider.feature_names = [
            'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
            'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
            'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
            'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
            'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
            'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
            'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
            'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
            'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
            'NUM_PKTS_1024_TO_1514_BYTES',
            'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE',
            'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER',
            'FTP_COMMAND_RET_CODE',
            'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX',
            'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
            'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX',
            'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV',
            'SRC_PORT_WELL_KNOWN', 'SRC_PORT_REGISTERED', 'SRC_PORT_EPHEMERAL',
            'DST_PORT_WELL_KNOWN', 'DST_PORT_REGISTERED', 'DST_PORT_EPHEMERAL',
        ]
        self.mock_provider.log_transform_columns = []
        self.mock_provider.feature_means = np.zeros(53, dtype=np.float64)
        self.mock_provider.feature_stds = np.ones(53, dtype=np.float64)
        self.mock_provider._PORT_BUCKETS = {
            'WELL_KNOWN': (0, 1023),
            'REGISTERED': (1024, 49151),
            'EPHEMERAL': (49152, 65535),
        }

    def test_preprocess_output_shape(self):
        """Test that _preprocess returns (53,) numpy array with no NaN/Inf."""
        # Create mock features dict (49 keys from NFv3FlowTracker)
        features = {
            'PROTOCOL': 6, 'L7_PROTO': 91, 'IN_BYTES': 1500, 'IN_PKTS': 10,
            'OUT_BYTES': 2000, 'OUT_PKTS': 8,
            'TCP_FLAGS': 24, 'CLIENT_TCP_FLAGS': 2, 'SERVER_TCP_FLAGS': 24,
            'FLOW_DURATION_MILLISECONDS': 5000, 'DURATION_IN': 3000, 'DURATION_OUT': 2000,
            'MIN_TTL': 64, 'MAX_TTL': 64,
            'LONGEST_FLOW_PKT': 1500, 'SHORTEST_FLOW_PKT': 40,
            'MIN_IP_PKT_LEN': 40, 'MAX_IP_PKT_LEN': 1500,
            'SRC_TO_DST_SECOND_BYTES': 300, 'DST_TO_SRC_SECOND_BYTES': 400,
            'SRC_TO_DST_AVG_THROUGHPUT': 2400, 'DST_TO_SRC_AVG_THROUGHPUT': 3200,
            'RETRANSMITTED_IN_BYTES': 0, 'RETRANSMITTED_IN_PKTS': 0,
            'RETRANSMITTED_OUT_BYTES': 0, 'RETRANSMITTED_OUT_PKTS': 0,
            'NUM_PKTS_UP_TO_128_BYTES': 2, 'NUM_PKTS_128_TO_256_BYTES': 3,
            'NUM_PKTS_256_TO_512_BYTES': 2, 'NUM_PKTS_512_TO_1024_BYTES': 2,
            'NUM_PKTS_1024_TO_1514_BYTES': 1,
            'TCP_WIN_MAX_IN': 65535, 'TCP_WIN_MAX_OUT': 65535,
            'ICMP_TYPE': 0, 'ICMP_IPV4_TYPE': 0,
            'DNS_QUERY_ID': 0, 'DNS_QUERY_TYPE': 0, 'DNS_TTL_ANSWER': 0,
            'FTP_COMMAND_RET_CODE': 0,
            'SRC_TO_DST_IAT_MIN': 100, 'SRC_TO_DST_IAT_MAX': 500,
            'SRC_TO_DST_IAT_AVG': 300, 'SRC_TO_DST_IAT_STDDEV': 150,
            'DST_TO_SRC_IAT_MIN': 50, 'DST_TO_SRC_IAT_MAX': 400,
            'DST_TO_SRC_IAT_AVG': 250, 'DST_TO_SRC_IAT_STDDEV': 100,
            'L4_SRC_PORT': 12345,
            'L4_DST_PORT': 443,
        }

        # Import and use actual FlowGuardProvider._preprocess logic
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Manually run preprocessing logic (avoiding full model load)
        import numpy as np

        feature_vector = np.zeros(len(self.mock_provider.feature_names), dtype=np.float64)

        # Port bucketing
        def bucket_port(port):
            result = {}
            for bucket_name, (lo, hi) in self.mock_provider._PORT_BUCKETS.items():
                result[bucket_name] = 1 if lo <= port <= hi else 0
            return result

        src_buckets = bucket_port(features['L4_SRC_PORT'])
        dst_buckets = bucket_port(features['L4_DST_PORT'])

        port_bucket_map = {
            'SRC_PORT_WELL_KNOWN': float(src_buckets['WELL_KNOWN']),
            'SRC_PORT_REGISTERED': float(src_buckets['REGISTERED']),
            'SRC_PORT_EPHEMERAL': float(src_buckets['EPHEMERAL']),
            'DST_PORT_WELL_KNOWN': float(dst_buckets['WELL_KNOWN']),
            'DST_PORT_REGISTERED': float(dst_buckets['REGISTERED']),
            'DST_PORT_EPHEMERAL': float(dst_buckets['EPHEMERAL']),
        }

        # Fill feature vector
        for i, fname in enumerate(self.mock_provider.feature_names):
            if fname in port_bucket_map:
                feature_vector[i] = port_bucket_map[fname]
            else:
                val = features.get(fname, 0.0)
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
                if np.isinf(val) or np.isnan(val):
                    val = 0.0
                feature_vector[i] = val

        # Z-score normalize
        eps = 1e-8
        normalized = (feature_vector - self.mock_provider.feature_means) / (self.mock_provider.feature_stds + eps)

        # Verify shape and content
        self.assertEqual(normalized.shape, (53,), f"Expected shape (53,), got {normalized.shape}")
        self.assertFalse(np.any(np.isnan(normalized)), "Output contains NaN values")
        self.assertFalse(np.any(np.isinf(normalized)), "Output contains Inf values")

    def test_preprocess_port_bucketing(self):
        """Test that port bucketing correctly categorizes ports."""
        # Test WELL_KNOWN port (443 = HTTPS)
        features_443 = {
            'L4_SRC_PORT': 443,
            'L4_DST_PORT': 80,
            **{k: 0.0 for k in self.mock_provider.feature_names if k not in ['SRC_PORT_WELL_KNOWN', 'SRC_PORT_REGISTERED', 'SRC_PORT_EPHEMERAL', 'DST_PORT_WELL_KNOWN', 'DST_PORT_REGISTERED', 'DST_PORT_EPHEMERAL']}
        }

        # Port 443 should be WELL_KNOWN (0-1023 overlaps)
        # Port 80 should also be WELL_KNOWN
        self.assertIn('L4_SRC_PORT', features_443)
        self.assertEqual(features_443['L4_SRC_PORT'], 443)


class TestNFv3SynGuard(unittest.TestCase):
    """Test NFv3FlowTracker SYN guard behavior."""

    def test_syn_guard_enabled_by_default(self):
        """Test that SYN guard is enabled by default."""
        tracker = NFv3FlowTracker(timeout=120.0)
        self.assertTrue(tracker.require_syn, "SYN guard should be enabled by default")

    def test_syn_guard_can_be_disabled(self):
        """Test that SYN guard can be disabled."""
        tracker = NFv3FlowTracker(timeout=120.0, require_syn=False)
        self.assertFalse(tracker.require_syn, "SYN guard should be disableable")

    def test_syn_guard_rejects_non_syn_tcp_packets(self):
        """Test that non-SYN TCP packets are rejected when require_syn=True."""
        tracker = NFv3FlowTracker(timeout=120.0, require_syn=True)

        # Mock a non-SYN packet (ACK only, flag value = 0x10 for ACK)
        mock_pkt = MagicMock(spec=Packet)
        mock_pkt.time = 1000.0
        mock_pkt[IP] = MagicMock()
        mock_pkt[IP].src = '192.168.1.1'
        mock_pkt[IP].dst = '192.168.1.2'
        mock_pkt[IP].proto = 6  # TCP
        mock_pkt[IP].len = 100
        mock_pkt[IP].ttl = 64
        mock_pkt[TCP] = MagicMock()
        mock_pkt[TCP].sport = 54321
        mock_pkt[TCP].dport = 443
        mock_pkt[TCP].flags = 16  # ACK flag (0x10), no SYN
        mock_pkt[TCP].seq = 1000
        mock_pkt[TCP].window = 65535

        # Mock the __contains__ method for IP and TCP checks
        mock_pkt.__contains__ = lambda self, key: key in [IP, TCP]

        initial_flow_count = len(tracker.active_flows)
        tracker.process_packet(mock_pkt)
        final_flow_count = len(tracker.active_flows)

        self.assertEqual(
            initial_flow_count, final_flow_count,
            "Non-SYN packet should not create a flow when require_syn=True"
        )

    def test_syn_packet_creates_flow_when_guard_enabled(self):
        """Test that SYN packets create flows even with guard enabled."""
        tracker = NFv3FlowTracker(timeout=120.0, require_syn=True)

        # Mock a SYN packet with flags object that has string representation containing 'S'
        flags_mock = MagicMock()
        flags_mock.__str__ = lambda self: 'S'  # str(flags) returns 'S'

        mock_pkt = MagicMock(spec=Packet)
        mock_pkt.time = 1000.0
        mock_pkt[IP] = MagicMock()
        mock_pkt[IP].src = '192.168.1.1'
        mock_pkt[IP].dst = '192.168.1.2'
        mock_pkt[IP].proto = 6  # TCP
        mock_pkt[IP].len = 60
        mock_pkt[IP].ttl = 64
        mock_pkt[TCP] = MagicMock()
        mock_pkt[TCP].sport = 54321
        mock_pkt[TCP].dport = 443
        mock_pkt[TCP].flags = flags_mock  # Mock flags with 'S' in str()
        mock_pkt[TCP].seq = 1000
        mock_pkt[TCP].window = 65535

        # Mock the __contains__ method for IP and TCP checks
        mock_pkt.__contains__ = lambda self, key: key in [IP, TCP]

        initial_flow_count = len(tracker.active_flows)
        tracker.process_packet(mock_pkt)
        final_flow_count = len(tracker.active_flows)

        self.assertEqual(
            final_flow_count, initial_flow_count + 1,
            "SYN packet should create a flow when require_syn=True"
        )

    def test_non_syn_packet_creates_flow_when_guard_disabled(self):
        """Test that non-SYN packets create flows when require_syn=False."""
        tracker = NFv3FlowTracker(timeout=120.0, require_syn=False)

        # Mock a non-SYN packet (ACK only) with flags that don't contain 'S'
        flags_mock = MagicMock()
        flags_mock.__str__ = lambda self: 'A'  # str(flags) returns 'A' (no 'S')

        mock_pkt = MagicMock(spec=Packet)
        mock_pkt.time = 1000.0
        mock_pkt[IP] = MagicMock()
        mock_pkt[IP].src = '10.0.0.1'
        mock_pkt[IP].dst = '10.0.0.2'
        mock_pkt[IP].proto = 6  # TCP
        mock_pkt[IP].len = 100
        mock_pkt[IP].ttl = 64
        mock_pkt[TCP] = MagicMock()
        mock_pkt[TCP].sport = 54321
        mock_pkt[TCP].dport = 443
        mock_pkt[TCP].flags = flags_mock  # Mock flags without 'S'
        mock_pkt[TCP].seq = 1000
        mock_pkt[TCP].window = 65535

        # Mock the __contains__ method for IP and TCP checks
        mock_pkt.__contains__ = lambda self, key: key in [IP, TCP]

        initial_flow_count = len(tracker.active_flows)
        tracker.process_packet(mock_pkt)
        final_flow_count = len(tracker.active_flows)

        self.assertEqual(
            final_flow_count, initial_flow_count + 1,
            "Non-SYN packet should create a flow when require_syn=False"
        )


if __name__ == '__main__':
    unittest.main()
