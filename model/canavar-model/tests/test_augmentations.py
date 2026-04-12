"""Tests for flow data augmentations."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.augmentations import FlowAugmentor
from src.data.features import get_input_dim


@pytest.fixture
def augmentor():
    config = {
        'feature_masking': {'enabled': True, 'mask_ratio': 0.2},
        'gaussian_noise': {'enabled': True, 'std': 0.05},
        'direction_swap': {'enabled': True, 'probability': 0.5},
        'temporal_jitter': {'enabled': True, 'jitter_ratio': 0.05},
    }
    return FlowAugmentor(config)


@pytest.fixture
def sample_batch():
    return torch.randn(32, get_input_dim())


class TestFlowAugmentor:

    def test_preserves_shape(self, augmentor, sample_batch):
        out = augmentor(sample_batch)
        assert out.shape == sample_batch.shape

    def test_two_views_are_different(self, augmentor, sample_batch):
        view1 = augmentor(sample_batch)
        view2 = augmentor(sample_batch)
        # They should almost certainly be different due to randomness
        assert not torch.allclose(view1, view2)

    def test_feature_masking_ratio(self, augmentor, sample_batch):
        """Approximately mask_ratio fraction of features should be zeroed."""
        augmentor.noise_enabled = False
        augmentor.swap_enabled = False
        augmentor.jitter_enabled = False

        out = augmentor(sample_batch)
        zero_frac = (out == 0).float().mean().item()
        # Should be roughly 0.2 (with some tolerance for features already at 0)
        assert 0.05 < zero_frac < 0.5

    def test_gaussian_noise_distribution(self):
        config = {
            'feature_masking': {'enabled': False},
            'gaussian_noise': {'enabled': True, 'std': 0.1},
            'direction_swap': {'enabled': False},
            'temporal_jitter': {'enabled': False},
        }
        aug = FlowAugmentor(config)
        x = torch.zeros(10000, get_input_dim())
        out = aug(x)
        # Output should have approximately std=0.1
        actual_std = out.std().item()
        assert 0.05 < actual_std < 0.15

    def test_no_augmentation(self):
        """With all augmentations disabled, output should equal input."""
        config = {
            'feature_masking': {'enabled': False},
            'gaussian_noise': {'enabled': False},
            'direction_swap': {'enabled': False},
            'temporal_jitter': {'enabled': False},
        }
        aug = FlowAugmentor(config)
        x = torch.randn(8, get_input_dim())
        out = aug(x)
        assert torch.allclose(out, x)
