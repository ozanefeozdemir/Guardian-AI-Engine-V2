"""
Integration tests: Model Provider & prediction pipeline.
Tests both PlaceholderModelProvider and LegacySklearnProvider.

Run: python -m pytest tests/test_engine_prediction.py -v
Run (integration): python -m pytest tests/test_engine_prediction.py -v -m integration
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'base_rf_2017.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_base.pkl')


# ---------- Model Provider Factory ----------

class TestModelProviderFactory:
    def test_get_placeholder_provider(self):
        from model_provider import get_model_provider
        provider = get_model_provider("placeholder")
        assert provider is not None
        assert provider.__class__.__name__ == "PlaceholderModelProvider"

    def test_get_legacy_provider(self):
        from model_provider import get_model_provider
        provider = get_model_provider("legacy")
        assert provider is not None
        assert provider.__class__.__name__ == "LegacySklearnProvider"

    def test_get_custom_provider(self):
        from model_provider import get_model_provider
        provider = get_model_provider("custom")
        assert provider is not None
        assert provider.__class__.__name__ == "CustomModelProvider"

    def test_invalid_provider_raises_error(self):
        from model_provider import get_model_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            get_model_provider("nonexistent")

    def test_env_var_fallback(self, monkeypatch):
        """When no name given, reads MODEL_PROVIDER env var."""
        monkeypatch.setenv("MODEL_PROVIDER", "placeholder")
        from model_provider import get_model_provider
        provider = get_model_provider(None)
        assert provider.__class__.__name__ == "PlaceholderModelProvider"


# ---------- Placeholder Provider ----------

class TestPlaceholderProvider:
    def test_load_succeeds(self):
        from model_provider import PlaceholderModelProvider
        provider = PlaceholderModelProvider()
        provider.load()
        assert provider.is_ready()

    def test_predict_returns_benign(self):
        from model_provider import PlaceholderModelProvider
        provider = PlaceholderModelProvider()
        provider.load()
        result = provider.predict({"Dst Port": 80})
        assert result["is_attack"] is False
        assert result["confidence"] == 0.0
        assert result["attack_type"] == "ModelNotLoaded"

    def test_predict_with_empty_features(self):
        from model_provider import PlaceholderModelProvider
        provider = PlaceholderModelProvider()
        provider.load()
        result = provider.predict({})
        assert result["is_attack"] is False

    def test_get_info(self):
        from model_provider import PlaceholderModelProvider
        provider = PlaceholderModelProvider()
        provider.load()
        info = provider.get_info()
        assert info["provider"] == "PlaceholderModelProvider"
        assert info["ready"] is True


# ---------- Custom Provider (Not Implemented) ----------

class TestCustomProvider:
    def test_load_raises_not_implemented(self):
        from model_provider import CustomModelProvider
        provider = CustomModelProvider()
        with pytest.raises(NotImplementedError):
            provider.load()

    def test_predict_raises_not_implemented(self):
        from model_provider import CustomModelProvider
        provider = CustomModelProvider()
        with pytest.raises(NotImplementedError):
            provider.predict({"Dst Port": 80})

    def test_is_not_ready_initially(self):
        from model_provider import CustomModelProvider
        provider = CustomModelProvider()
        assert provider.is_ready() is False


# ---------- Legacy Provider (Integration) ----------

pytestmark_integration = pytest.mark.integration


class TestLegacyProviderLoading:
    @pytest.mark.integration
    def test_model_file_exists(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"Model file not found: {MODEL_PATH}")
        assert os.path.exists(MODEL_PATH)

    @pytest.mark.integration
    def test_scaler_file_exists(self):
        if not os.path.exists(SCALER_PATH):
            pytest.skip(f"Scaler file not found: {SCALER_PATH}")
        assert os.path.exists(SCALER_PATH)

    @pytest.mark.integration
    def test_legacy_provider_loads(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not found")
        from model_provider import LegacySklearnProvider
        provider = LegacySklearnProvider()
        provider.load()
        assert provider.is_ready()

    @pytest.mark.integration
    def test_model_has_warm_start(self):
        """Model should support warm_start for online adaptation."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not found")
        from model_provider import LegacySklearnProvider
        provider = LegacySklearnProvider()
        provider.load()
        assert getattr(provider.model, 'warm_start', False) is True


class TestLegacyProviderPrediction:
    @pytest.fixture
    def legacy_provider(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not found")
        from model_provider import LegacySklearnProvider
        provider = LegacySklearnProvider()
        provider.load()
        return provider

    @pytest.mark.integration
    def test_prediction_returns_required_keys(self, legacy_provider):
        result = legacy_provider.predict({"Dst Port": 80, "Tot Fwd Pkts": 10, "Flow Byts/s": 500.0})
        assert "is_attack" in result
        assert "confidence" in result
        assert "attack_type" in result

    @pytest.mark.integration
    def test_prediction_confidence_range(self, legacy_provider):
        result = legacy_provider.predict({"Dst Port": 80, "Tot Fwd Pkts": 10})
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.integration
    def test_prediction_returns_bool_attack(self, legacy_provider):
        result = legacy_provider.predict({"Dst Port": 443})
        assert isinstance(result["is_attack"], bool)

    @pytest.mark.integration
    def test_prediction_with_all_zeros(self, legacy_provider):
        result = legacy_provider.predict({})
        assert isinstance(result["is_attack"], bool)

    @pytest.mark.integration
    def test_prediction_with_extreme_values(self, legacy_provider):
        result = legacy_provider.predict({
            "Dst Port": 65535,
            "Tot Fwd Pkts": 999999,
            "Flow Byts/s": 1e12,
        })
        assert isinstance(result["is_attack"], bool)


# ---------- Engine Process Packet with Provider ----------

class TestProcessPacketWithProvider:
    def test_process_packet_with_placeholder(self):
        """Engine should work with PlaceholderProvider (no real model needed)."""
        try:
            import redis as sync_redis
            r = sync_redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
        except Exception:
            pytest.skip("Redis not available")

        import json
        from analyze_engine import TrafficEngine
        from model_provider import PlaceholderModelProvider

        test_queue = "test_engine_alerts"
        r.delete(test_queue)

        engine = TrafficEngine(mode="simulation")
        engine.redis_client = r

        # Inject placeholder provider
        provider = PlaceholderModelProvider()
        provider.load()
        engine.provider = provider

        import analyze_engine
        original_queue = analyze_engine.ALERT_QUEUE
        analyze_engine.ALERT_QUEUE = test_queue

        try:
            engine.process_packet(
                features={"Dst Port": 80, "Tot Fwd Pkts": 10},
                source_meta="test_ip"
            )

            raw = r.lpop(test_queue)
            assert raw is not None
            data = json.loads(raw)
            assert data["source"] == "test_ip"
            assert data["is_attack"] is False
            assert data["confidence"] == 0.0
            assert data["attack_type"] == "ModelNotLoaded"
        finally:
            analyze_engine.ALERT_QUEUE = original_queue
            r.delete(test_queue)
            r.close()

    @pytest.mark.integration
    def test_process_packet_with_legacy_provider(self):
        """Engine should work with LegacyProvider."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model file not found")

        try:
            import redis as sync_redis
            r = sync_redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
        except Exception:
            pytest.skip("Redis not available")

        import json
        from analyze_engine import TrafficEngine
        from model_provider import LegacySklearnProvider

        test_queue = "test_engine_alerts"
        r.delete(test_queue)

        engine = TrafficEngine(mode="simulation")
        engine.redis_client = r

        provider = LegacySklearnProvider()
        provider.load()
        engine.provider = provider

        import analyze_engine
        original_queue = analyze_engine.ALERT_QUEUE
        analyze_engine.ALERT_QUEUE = test_queue

        try:
            engine.process_packet(
                features={"Dst Port": 80, "Tot Fwd Pkts": 10},
                source_meta="test_ip"
            )

            raw = r.lpop(test_queue)
            assert raw is not None
            data = json.loads(raw)
            assert "timestamp" in data
            assert data["source"] == "test_ip"
            assert "is_attack" in data
            assert "confidence" in data
        finally:
            analyze_engine.ALERT_QUEUE = original_queue
            r.delete(test_queue)
            r.close()
