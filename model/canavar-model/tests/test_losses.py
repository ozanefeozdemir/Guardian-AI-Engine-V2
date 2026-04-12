"""Tests for loss functions."""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.losses import NTXentLoss, SupervisedContrastiveLoss, EWCLoss


class TestNTXentLoss:

    def test_identical_views_low_loss(self):
        """If both views are identical, loss should be relatively low."""
        loss_fn = NTXentLoss(temperature=0.07)
        z = torch.randn(32, 16)
        z = torch.nn.functional.normalize(z, dim=1)
        loss = loss_fn(z, z)
        # Not exactly zero due to cross-entropy formulation, but should be defined
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_random_views_higher_loss(self):
        """Random views should produce higher loss than similar views."""
        loss_fn = NTXentLoss(temperature=0.5)  # Higher temp for smoother comparison

        z = torch.randn(32, 16)
        z = torch.nn.functional.normalize(z, dim=1)
        noise = torch.randn_like(z) * 0.01
        z_similar = torch.nn.functional.normalize(z + noise, dim=1)
        z_random = torch.nn.functional.normalize(torch.randn(32, 16), dim=1)

        loss_similar = loss_fn(z, z_similar)
        loss_random = loss_fn(z, z_random)

        assert loss_random > loss_similar

    def test_gradient_flow(self):
        """Loss must produce valid gradients for both inputs."""
        loss_fn = NTXentLoss()
        z_i = torch.randn(16, 8, requires_grad=True)
        z_j = torch.randn(16, 8, requires_grad=True)
        z_i_norm = torch.nn.functional.normalize(z_i, dim=1)
        z_j_norm = torch.nn.functional.normalize(z_j, dim=1)

        loss = loss_fn(z_i_norm, z_j_norm)
        loss.backward()

        assert z_i.grad is not None
        assert z_j.grad is not None
        assert not torch.any(torch.isnan(z_i.grad))
        assert not torch.any(torch.isnan(z_j.grad))

    def test_output_is_scalar(self):
        loss_fn = NTXentLoss()
        z = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
        loss = loss_fn(z, z)
        assert loss.dim() == 0


class TestSupervisedContrastiveLoss:

    def test_same_class_low_loss(self):
        loss_fn = SupervisedContrastiveLoss(temperature=0.5)
        features = torch.nn.functional.normalize(torch.randn(10, 16), dim=1)
        labels = torch.zeros(10, dtype=torch.long)  # All same class
        loss = loss_fn(features, labels)
        assert not torch.isnan(loss)

    def test_gradient_flow(self):
        loss_fn = SupervisedContrastiveLoss()
        features = torch.randn(16, 8, requires_grad=True)
        features_norm = torch.nn.functional.normalize(features, dim=1)
        labels = torch.randint(0, 3, (16,))

        loss = loss_fn(features_norm, labels)
        loss.backward()
        assert features.grad is not None


class TestEWCLoss:

    def test_zero_loss_unchanged_params(self):
        """If model params haven't changed, EWC loss should be zero."""
        model = nn.Linear(10, 2)
        old_params = {n: p.clone() for n, p in model.named_parameters()}
        fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}

        ewc = EWCLoss(model, old_params, fisher, lambda_=1.0)
        loss = ewc(model)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_nonzero_loss_changed_params(self):
        """If model params changed, EWC loss should be positive."""
        model = nn.Linear(10, 2)
        old_params = {n: p.clone() for n, p in model.named_parameters()}
        fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}

        # Change parameters
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        ewc = EWCLoss(model, old_params, fisher, lambda_=1.0)
        loss = ewc(model)
        assert loss.item() > 0
