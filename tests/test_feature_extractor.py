"""
Unit tests for backend/feature_extractor.py
No external dependencies (Redis, DB, model files) required.
"""
import numpy as np
import pytest

from feature_extractor import FeatureExtractor, MAPPING, ORDERED_FEATURES


# ──────────────────────────────────────────────
#  Transform with short (CSV) keys
# ──────────────────────────────────────────────
class TestTransformShortKeys:
    def test_returns_numpy_array(self, mock_scaler, sample_features_short_keys):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform(sample_features_short_keys)
        assert isinstance(result, np.ndarray)

    def test_output_shape(self, mock_scaler, sample_features_short_keys):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform(sample_features_short_keys)
        assert result.shape == (1, len(ORDERED_FEATURES))

    def test_mapped_values_are_present(self, mock_scaler):
        """Dst Port=80 should map to Destination Port and appear in output."""
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform({"Dst Port": 80})
        idx = ORDERED_FEATURES.index("Destination Port")
        assert result[0, idx] == 80.0


# ──────────────────────────────────────────────
#  Transform with long (mapped) keys
# ──────────────────────────────────────────────
class TestTransformLongKeys:
    def test_output_shape(self, mock_scaler, sample_features_long_keys):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform(sample_features_long_keys)
        assert result.shape == (1, len(ORDERED_FEATURES))

    def test_values_preserved(self, mock_scaler):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform({"Destination Port": 443})
        idx = ORDERED_FEATURES.index("Destination Port")
        assert result[0, idx] == 443.0


# ──────────────────────────────────────────────
#  Missing features filled with zero
# ──────────────────────────────────────────────
class TestMissingFeatures:
    def test_empty_dict_gives_all_zeros(self, mock_scaler):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform({})
        assert result.shape == (1, len(ORDERED_FEATURES))
        assert np.all(result == 0.0)

    def test_partial_dict_fills_missing(self, mock_scaler):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform({"Dst Port": 80})
        # All columns except Destination Port should be 0
        idx = ORDERED_FEATURES.index("Destination Port")
        non_port = np.delete(result[0], idx)
        assert np.all(non_port == 0.0)


# ──────────────────────────────────────────────
#  Inf / NaN handling
# ──────────────────────────────────────────────
class TestInfNanHandling:
    def test_inf_replaced_with_zero(self, mock_scaler):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform({"Dst Port": float("inf")})
        idx = ORDERED_FEATURES.index("Destination Port")
        assert result[0, idx] == 0.0

    def test_negative_inf_replaced_with_zero(self, mock_scaler):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform({"Dst Port": float("-inf")})
        idx = ORDERED_FEATURES.index("Destination Port")
        assert result[0, idx] == 0.0

    def test_nan_replaced_with_zero(self, mock_scaler):
        ext = FeatureExtractor(mock_scaler)
        result = ext.transform({"Dst Port": float("nan")})
        idx = ORDERED_FEATURES.index("Destination Port")
        assert result[0, idx] == 0.0


# ──────────────────────────────────────────────
#  Column ordering & MAPPING consistency
# ──────────────────────────────────────────────
class TestColumnOrdering:
    def test_ordered_features_matches_mapping_values(self):
        assert ORDERED_FEATURES == list(MAPPING.values())

    def test_no_duplicate_features(self):
        assert len(ORDERED_FEATURES) == len(set(ORDERED_FEATURES))

    def test_feature_count(self):
        assert len(ORDERED_FEATURES) == 48
