"""
Shared fixtures for Guardian AI Engine V2 tests.
"""
import sys
import os
import pytest
import numpy as np

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


# ---------- Markers ----------
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests requiring Redis/PostgreSQL (deselect with '-m \"not integration\"')"
    )


# ---------- Mock Scaler ----------
class MockScaler:
    """A scaler that returns input unchanged (identity transform)."""
    def transform(self, X):
        return np.array(X, dtype='float32')

    def fit_transform(self, X):
        return np.array(X, dtype='float32')


@pytest.fixture
def mock_scaler():
    return MockScaler()


# ---------- Sample Feature Dict ----------
@pytest.fixture
def sample_features_short_keys():
    """Features using CIC-IDS short column names (pre-mapping)."""
    return {
        "Dst Port": 80,
        "Tot Fwd Pkts": 10,
        "Tot Bwd Pkts": 5,
        "Flow Byts/s": 500.0,
        "Flow Pkts/s": 20.0,
    }


@pytest.fixture
def sample_features_long_keys():
    """Features using mapped (long) column names."""
    return {
        "Destination Port": 80,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 5,
        "Flow Bytes/s": 500.0,
        "Flow Packets/s": 20.0,
    }


@pytest.fixture
def sample_alert_data():
    """A complete alert dict as produced by analyze_engine."""
    import time
    return {
        "timestamp": time.time(),
        "source": "192.168.1.100",
        "is_attack": True,
        "confidence": 0.95,
        "attack_type": "DDoS",
        "original_features": {"Destination Port": 80, "Total Fwd Packets": 10},
    }
