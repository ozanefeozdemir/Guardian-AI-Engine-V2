"""Phase 1: Simple MLP baseline for binary/multiclass flow classification."""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Multi-layer perceptron baseline.

    Args:
        input_dim: Number of input features (default: 49)
        hidden_dims: List of hidden layer sizes (default: [256, 128, 64])
        num_classes: Number of output classes (default: 2 for binary)
        dropout: Dropout rate (default: 0.3)
        activation: Activation function (default: "relu")
    """

    def __init__(self, input_dim=49, hidden_dims=None, num_classes=2,
                 dropout=0.3, activation="relu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extract features before classification head."""
        return self.backbone(x)
