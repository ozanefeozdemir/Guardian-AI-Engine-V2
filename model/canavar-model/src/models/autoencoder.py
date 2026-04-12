"""
Autoencoder for energy-based anomaly scoring (Bertoli stacking).
Trained on benign-only traffic. Reconstruction error = anomaly score.
"""

import torch
import torch.nn as nn


class FlowAutoencoder(nn.Module):
    """
    Symmetric autoencoder for flow features.

    Args:
        input_dim: Number of input features
        hidden_dims: Encoder hidden dimensions (decoder mirrors them)
    """

    def __init__(self, input_dim=49, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]

        # Encoder
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers = []
        decoder_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev = hidden_dims[-1]
        for i, h in enumerate(decoder_dims):
            decoder_layers.append(nn.Linear(prev, h))
            if i < len(decoder_dims) - 1:
                decoder_layers.append(nn.ReLU(inplace=True))
            prev = h
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Returns (reconstruction, latent)."""
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def energy_score(self, x):
        """Compute reconstruction error (MSE per sample) as anomaly score."""
        recon, _ = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)
