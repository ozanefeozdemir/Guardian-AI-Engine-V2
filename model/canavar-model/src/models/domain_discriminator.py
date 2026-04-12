"""
Domain discriminator for domain adversarial training.
Uses gradient reversal to make encoder domain-invariant.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverses gradients during backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """
    Classifies which dataset/domain an embedding comes from.
    Used with gradient reversal to encourage domain-invariant representations.
    """

    def __init__(self, input_dim=128, hidden_dims=None, num_domains=4):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]

        self.grl = GradientReversalLayer()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, num_domains))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x, lambda_=1.0):
        self.grl.lambda_ = lambda_
        x = self.grl(x)
        return self.classifier(x)
