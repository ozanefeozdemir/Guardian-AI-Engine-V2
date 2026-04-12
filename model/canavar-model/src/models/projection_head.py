"""Projection head for contrastive learning. Maps encoder output to lower-dim space."""

import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning (SimCLR style).
    Maps encoder embeddings to a lower-dimensional space where contrastive loss is applied.

    Architecture: Linear -> BatchNorm -> ReLU -> Linear -> L2-normalize
    """

    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Args: x: (batch_size, input_dim) — encoder output
        Returns: (batch_size, output_dim) — projected embedding (L2-normalized)
        """
        z = self.net(x)
        return F.normalize(z, dim=1)
