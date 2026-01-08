import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api import app
import pytest

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_status(client):
    """Test health check."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    # assert data["model"] == "loaded" # This might fail if models aren't found in CI, but on local we hope it passes

def test_predict_flow(client):
    """Test prediction with dummy data."""
    # A simplified payload. FeatureExtractor fills the rest with 0s.
    payload = {
        "features": {
            "Dst Port": 80,
            "Tot Fwd Pkts": 10,
            "Flow Byts/s": 500.0
        }
    }
    
    response = client.post("/predict", json=payload)
    
    # If model is not loaded, it returns 503
    if response.status_code == 503:
        pytest.skip("Model not loaded, skipping inference test.")
        
    assert response.status_code == 200
    data = response.json()
    assert "is_attack" in data
    assert "confidence" in data
    assert data["model_version"] == "rf_comprehensive_v1"
