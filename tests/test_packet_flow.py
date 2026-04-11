"""
Unit tests for backend/packet_flow.py — CICFlowTracker
Scapy mock paketleri ile test edilir, Redis/model gerektirmez.

Çalıştır: python -m pytest tests/test_packet_flow.py -v
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from scapy.all import IP, TCP, UDP, Ether
from packet_flow import CICFlowTracker

# cols.txt'deki tüm feature isimleri (Label hariç)
EXPECTED_FEATURES = [
    'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration',
    'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len',
    'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
    'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]


# ──────────────────────────────────────────────
#  Helper: Mock paket oluşturma
# ──────────────────────────────────────────────

def make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", sport=12345, dport=80,
                 flags="A", payload=b"", ts=1000.0, window=8192):
    """Belirli parametrelerle mock TCP paketi oluştur."""
    pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags, window=window) / payload
    pkt.time = ts
    return pkt


def make_udp_pkt(src="10.0.0.1", dst="10.0.0.2", sport=12345, dport=53,
                 payload=b"", ts=1000.0):
    """Mock UDP paketi oluştur."""
    pkt = IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / payload
    pkt.time = ts
    return pkt


# ──────────────────────────────────────────────
#  Feature Completeness
# ──────────────────────────────────────────────

class TestFeatureCompleteness:
    """extract_features'ın 79 feature'ın tamamını döndürdüğünü doğrula."""

    def test_all_features_present(self):
        """TCP SYN+FIN flow'unda tüm feature key'leri mevcut olmalı."""
        tracker = CICFlowTracker(timeout=120.0)

        # SYN → ACK → FIN (flow kapanır)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        tracker.process_packet(make_tcp_pkt(flags="A", ts=1000.1))
        result = tracker.process_packet(make_tcp_pkt(flags="FA", ts=1000.2))

        assert result is not None, "FIN bayrağıyla flow kapanmalı ve features dönmeli"
        for feat in EXPECTED_FEATURES:
            assert feat in result, f"Eksik feature: {feat}"

    def test_feature_count(self):
        """Tam 79 feature olmalı."""
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        result = tracker.process_packet(make_tcp_pkt(flags="F", ts=1000.1))
        assert len(result) == 84

    def test_udp_flow_features(self):
        """UDP flow'unda da 79 feature olmalı (flush ile)."""
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_udp_pkt(ts=1000.0))
        tracker.process_packet(make_udp_pkt(ts=1000.1))
        tracker.flush_all()
        # flush_all sonrası active_flows boş olmalı
        assert len(tracker.active_flows) == 0


# ──────────────────────────────────────────────
#  Flow Direction
# ──────────────────────────────────────────────

class TestFlowDirection:
    """Fwd ve Bwd paketlerin doğru sayıldığını doğrula."""

    def test_fwd_bwd_packet_counts(self):
        tracker = CICFlowTracker(timeout=120.0)

        # 3 Fwd paket
        tracker.process_packet(make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", flags="S", ts=1000.0))
        tracker.process_packet(make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", flags="A", ts=1000.1))
        tracker.process_packet(make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", flags="PA", ts=1000.2))

        # 2 Bwd paket
        tracker.process_packet(make_tcp_pkt(src="10.0.0.2", dst="10.0.0.1", sport=80, dport=12345, flags="A", ts=1000.3))
        result = tracker.process_packet(
            make_tcp_pkt(src="10.0.0.2", dst="10.0.0.1", sport=80, dport=12345, flags="FA", ts=1000.4)
        )

        assert result is not None
        assert result['Tot Fwd Pkts'] == 3
        assert result['Tot Bwd Pkts'] == 2

    def test_dst_port_and_protocol(self):
        """Dst Port ve Protocol doğru kaydedilmeli."""
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(dport=443, flags="S", ts=1000.0))
        result = tracker.process_packet(make_tcp_pkt(dport=443, flags="F", ts=1000.1))

        assert result['Dst Port'] == 443
        assert result['Protocol'] == 6  # TCP


# ──────────────────────────────────────────────
#  Timeout Expiration
# ──────────────────────────────────────────────

class TestTimeoutExpiration:
    """Timeout mekanizmasının çalıştığını doğrula."""

    def test_timeout_closes_flow(self):
        tracker = CICFlowTracker(timeout=10.0)

        # Flow başlat (t=1000)
        tracker.process_packet(make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", flags="A", ts=1000.0))
        assert len(tracker.active_flows) == 1

        # 15 saniye sonra farklı bir flow'dan paket gelsin → timeout tetiklenir
        tracker.process_packet(
            make_tcp_pkt(src="10.0.0.3", dst="10.0.0.4", sport=5555, dport=8080, flags="S", ts=1015.0)
        )

        # İlk flow timeout'tan dolayı kapanmış olmalı
        original_key = [k for k in tracker.active_flows.keys() if "10.0.0.1" in k]
        assert len(original_key) == 0, "Timeout olan flow kapanmalıydı"

    def test_flush_all_closes_everything(self):
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", flags="S", ts=1000.0))
        tracker.process_packet(make_tcp_pkt(src="10.0.0.3", dst="10.0.0.4", sport=5555, dport=8080, flags="S", ts=1000.0))

        assert len(tracker.active_flows) == 2
        tracker.flush_all()
        assert len(tracker.active_flows) == 0


# ──────────────────────────────────────────────
#  FIN/RST Closure
# ──────────────────────────────────────────────

class TestTcpClosure:
    """TCP FIN/RST bayrağıyla flow kapanması."""

    def test_fin_closes_flow(self):
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        assert len(tracker.active_flows) == 1

        tracker.process_packet(make_tcp_pkt(flags="FA", ts=1000.1))
        assert len(tracker.active_flows) == 0

    def test_rst_closes_flow(self):
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        tracker.process_packet(make_tcp_pkt(flags="R", ts=1000.1))
        assert len(tracker.active_flows) == 0


# ──────────────────────────────────────────────
#  Callback Invocation
# ──────────────────────────────────────────────

class TestCallback:
    """on_flow_ready callback'inin çağrıldığını doğrula."""

    def test_callback_called_on_fin(self):
        results = []
        def cb(features, src_ip, dst_ip):
            results.append({'features': features, 'src': src_ip, 'dst': dst_ip})

        tracker = CICFlowTracker(timeout=120.0, on_flow_ready=cb)
        tracker.process_packet(make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", flags="S", ts=1000.0))
        tracker.process_packet(make_tcp_pkt(src="10.0.0.1", dst="10.0.0.2", flags="FA", ts=1000.1))

        assert len(results) == 1
        assert results[0]['src'] == "10.0.0.1"
        assert results[0]['dst'] == "10.0.0.2"
        assert 'Flow Duration' in results[0]['features']

    def test_callback_called_on_flush(self):
        results = []
        def cb(features, src_ip, dst_ip):
            results.append(features)

        tracker = CICFlowTracker(timeout=120.0, on_flow_ready=cb)
        tracker.process_packet(make_udp_pkt(ts=1000.0))
        tracker.flush_all()

        assert len(results) == 1
        assert results[0]['Protocol'] == 17  # UDP

    def test_no_callback_if_none(self):
        """Callback None ise hata vermemeli."""
        tracker = CICFlowTracker(timeout=120.0, on_flow_ready=None)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        tracker.process_packet(make_tcp_pkt(flags="F", ts=1000.1))
        assert len(tracker.active_flows) == 0


# ──────────────────────────────────────────────
#  Feature Values Sanity
# ──────────────────────────────────────────────

class TestFeatureValues:
    """Feature değerlerinin mantıklı olduğunu doğrula."""

    def test_flow_duration_positive(self):
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        result = tracker.process_packet(make_tcp_pkt(flags="F", ts=1001.0))

        # 1 saniye = 1_000_000 mikrosaniye
        assert result['Flow Duration'] == pytest.approx(1_000_000.0, rel=0.01)

    def test_flag_counts(self):
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        tracker.process_packet(make_tcp_pkt(flags="PA", ts=1000.1))
        result = tracker.process_packet(make_tcp_pkt(flags="FA", ts=1000.2))

        assert result['SYN Flag Cnt'] == 1
        assert result['PSH Flag Cnt'] == 1
        assert result['ACK Flag Cnt'] == 2  # PA + FA
        assert result['FIN Flag Cnt'] == 1

    def test_no_nan_or_inf(self):
        """Feature'larda NaN veya Inf olmamalı."""
        tracker = CICFlowTracker(timeout=120.0)
        tracker.process_packet(make_tcp_pkt(flags="S", ts=1000.0))
        result = tracker.process_packet(make_tcp_pkt(flags="F", ts=1000.0))  # aynı zaman

        for key, val in result.items():
            if isinstance(val, float):
                assert not np.isnan(val), f"{key} NaN olmamalı"
                assert not np.isinf(val), f"{key} Inf olmamalı"
