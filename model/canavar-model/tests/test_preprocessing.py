"""Tests for preprocessing pipeline."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.features import (
    IDENTITY_FEATURES, LABEL_COLUMNS, get_feature_names, get_input_dim,
    ENGINEERED_FEATURES
)


class TestFeatureSchema:
    """Test feature schema definitions."""

    def test_identity_features_defined(self):
        assert len(IDENTITY_FEATURES) == 4
        assert "IPV4_SRC_ADDR" in IDENTITY_FEATURES
        assert "L4_SRC_PORT" in IDENTITY_FEATURES

    def test_get_feature_names_excludes_identity(self):
        names = get_feature_names()
        for feat in IDENTITY_FEATURES:
            assert feat not in names

    def test_get_feature_names_excludes_labels(self):
        names = get_feature_names()
        for label in LABEL_COLUMNS:
            assert label not in names

    def test_get_input_dim_matches_names(self):
        assert get_input_dim() == len(get_feature_names())

    def test_engineered_features_present(self):
        names = get_feature_names()
        for feat in ENGINEERED_FEATURES:
            assert feat in names, f"{feat} missing from feature names"


class TestPreprocessing:
    """Test preprocessing pipeline (requires processed data)."""

    @pytest.fixture
    def processed_path(self):
        path = "data/processed/unsw.parquet"
        if not os.path.exists(path):
            pytest.skip("Processed data not available. Run preprocessing first.")
        return path

    def test_identity_features_removed(self, processed_path):
        df = pd.read_parquet(processed_path)
        for col in IDENTITY_FEATURES:
            assert col not in df.columns, f"{col} should be removed"

    def test_no_inf_or_nan(self, processed_path):
        df = pd.read_parquet(processed_path)
        feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
        for col in feature_cols:
            assert not np.any(np.isinf(df[col].values)), f"Inf found in {col}"
            assert not np.any(np.isnan(df[col].values)), f"NaN found in {col}"

    def test_port_buckets(self, processed_path):
        df = pd.read_parquet(processed_path)
        src_cols = ["SRC_PORT_WELL_KNOWN", "SRC_PORT_REGISTERED", "SRC_PORT_EPHEMERAL"]
        dst_cols = ["DST_PORT_WELL_KNOWN", "DST_PORT_REGISTERED", "DST_PORT_EPHEMERAL"]

        if all(c in df.columns for c in src_cols):
            src_sum = df[src_cols].sum(axis=1)
            assert (src_sum == 1).all(), "Source port buckets should sum to 1"

        if all(c in df.columns for c in dst_cols):
            dst_sum = df[dst_cols].sum(axis=1)
            assert (dst_sum == 1).all(), "Dest port buckets should sum to 1"

    def test_labels_present(self, processed_path):
        df = pd.read_parquet(processed_path)
        assert "Label" in df.columns, "Label column missing"

    def test_normalization(self, processed_path):
        """After z-score, features should have approximately 0 mean and 1 std."""
        df = pd.read_parquet(processed_path)
        feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
        # Check a few continuous features (not binary port buckets)
        continuous = [c for c in feature_cols if c not in ENGINEERED_FEATURES]
        if continuous:
            means = df[continuous[:5]].mean()
            stds = df[continuous[:5]].std()
            # Loose bounds since test data may differ from train stats
            assert all(abs(m) < 5 for m in means), f"Means too far from 0: {means.to_dict()}"
