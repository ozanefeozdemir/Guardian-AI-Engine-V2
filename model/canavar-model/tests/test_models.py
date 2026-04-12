"""Tests for model architectures."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mlp_baseline import MLPBaseline
from src.models.transformer_encoder import FlowTransformerEncoder
from src.models.projection_head import ProjectionHead
from src.models.classification_head import ClassificationHead
from src.models.autoencoder import FlowAutoencoder
from src.models.domain_discriminator import DomainDiscriminator, GradientReversalLayer


class TestMLPBaseline:

    def test_forward_shape(self):
        model = MLPBaseline(input_dim=49, num_classes=2)
        x = torch.randn(16, 49)
        out = model(x)
        assert out.shape == (16, 2)

    def test_multiclass(self):
        model = MLPBaseline(input_dim=49, num_classes=10)
        x = torch.randn(8, 49)
        out = model(x)
        assert out.shape == (8, 10)

    def test_get_features(self):
        model = MLPBaseline(input_dim=49, hidden_dims=[128, 64])
        x = torch.randn(8, 49)
        feat = model.get_features(x)
        assert feat.shape == (8, 64)


class TestTransformerEncoder:

    def test_forward_shape(self):
        model = FlowTransformerEncoder(input_dim=49, model_dim=128)
        x = torch.randn(8, 49)
        out = model(x)
        assert out.shape == (8, 128)

    def test_different_dims(self):
        model = FlowTransformerEncoder(input_dim=30, model_dim=64, num_heads=2, num_layers=2)
        x = torch.randn(4, 30)
        out = model(x)
        assert out.shape == (4, 64)

    def test_gradient_flow(self):
        model = FlowTransformerEncoder(input_dim=49, model_dim=128)
        x = torch.randn(4, 49, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestProjectionHead:

    def test_forward_shape(self):
        head = ProjectionHead(input_dim=128, hidden_dim=64, output_dim=32)
        x = torch.randn(8, 128)
        out = head(x)
        assert out.shape == (8, 32)

    def test_output_normalized(self):
        head = ProjectionHead(input_dim=128, output_dim=32)
        x = torch.randn(8, 128)
        out = head(x)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)


class TestClassificationHead:

    def test_forward_shape(self):
        head = ClassificationHead(input_dim=128, num_classes=2)
        x = torch.randn(8, 128)
        out = head(x)
        assert out.shape == (8, 2)


class TestAutoencoder:

    def test_reconstruction_shape(self):
        ae = FlowAutoencoder(input_dim=49)
        x = torch.randn(8, 49)
        recon, z = ae(x)
        assert recon.shape == x.shape

    def test_energy_score_shape(self):
        ae = FlowAutoencoder(input_dim=49)
        x = torch.randn(8, 49)
        scores = ae.energy_score(x)
        assert scores.shape == (8,)

    def test_energy_score_positive(self):
        ae = FlowAutoencoder(input_dim=49)
        x = torch.randn(8, 49)
        scores = ae.energy_score(x)
        assert (scores >= 0).all()


class TestDomainDiscriminator:

    def test_forward_shape(self):
        disc = DomainDiscriminator(input_dim=128, num_domains=4)
        x = torch.randn(8, 128)
        out = disc(x)
        assert out.shape == (8, 4)

    def test_gradient_reversal(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 10, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # Gradient should be negated
        assert torch.allclose(x.grad, -torch.ones_like(x))
