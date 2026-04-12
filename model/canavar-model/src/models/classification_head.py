"""Reusable classification head that attaches to any encoder backbone."""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head MLP. Attaches to encoder output.

    Args:
        input_dim: Encoder output dimension
        hidden_dims: Hidden layer sizes (default: [64])
        num_classes: Number of output classes
        dropout: Dropout rate
    """

    def __init__(self, input_dim=128, hidden_dims=None, num_classes=2, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
