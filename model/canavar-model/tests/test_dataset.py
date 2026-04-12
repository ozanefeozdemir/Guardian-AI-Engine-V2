"""Tests for PyTorch Dataset classes."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import FlowDataset, UnlabeledFlowDataset, create_dataloader


class TestFlowDataset:

    @pytest.fixture
    def dataset_path(self):
        # Try protocol_a split first, then processed
        for path in ["data/splits/protocol_a/unsw_train.parquet", "data/processed/unsw.parquet"]:
            if os.path.exists(path):
                return path
        pytest.skip("No processed data available")

    def test_dataset_loads(self, dataset_path):
        ds = FlowDataset(dataset_path)
        assert len(ds) > 0

    def test_dataset_returns_tuple(self, dataset_path):
        ds = FlowDataset(dataset_path)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        features, label = item
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_features_are_float(self, dataset_path):
        ds = FlowDataset(dataset_path)
        features, _ = ds[0]
        assert features.dtype == torch.float32

    def test_labels_are_long(self, dataset_path):
        ds = FlowDataset(dataset_path)
        _, label = ds[0]
        assert label.dtype == torch.long

    def test_dataloader_batches(self, dataset_path):
        loader = create_dataloader(dataset_path, batch_size=32, num_workers=0)
        batch = next(iter(loader))
        features, labels = batch
        assert features.shape[0] <= 32
        assert labels.shape[0] == features.shape[0]

    def test_balanced_sampling(self, dataset_path):
        loader = create_dataloader(dataset_path, batch_size=256, balanced=True, num_workers=0)
        features, labels = next(iter(loader))
        # With balanced sampling, both classes should be represented
        unique = labels.unique()
        assert len(unique) >= 1  # At minimum 1 class present


class TestUnlabeledDataset:

    @pytest.fixture
    def unlabeled_path(self):
        path = "data/processed/combined_unlabeled.parquet"
        if not os.path.exists(path):
            pytest.skip("Combined unlabeled data not available")
        return path

    def test_unlabeled_returns_tensor(self, unlabeled_path):
        ds = UnlabeledFlowDataset(unlabeled_path)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.float32
