"""Phase 2: Transformer encoder backbone for flow feature embeddings."""

import torch
import torch.nn as nn


class FlowTransformerEncoder(nn.Module):
    """
    Transformer encoder that maps flow feature vectors to fixed-size embeddings.

    Architecture: Input(49) -> Linear(1, model_dim) per feature -> [CLS] prepended
    -> TransformerEncoder -> [CLS] output -> LayerNorm -> model_dim embedding.

    Each scalar feature is treated as a token, projected to model_dim. A learnable
    [CLS] token is prepended and its output representation is used as the
    fixed-size embedding.

    Args:
        input_dim: Number of input features (default: 49)
        model_dim: Transformer hidden dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer encoder layers (default: 4)
        feedforward_dim: Feed-forward network dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, input_dim=49, model_dim=128, num_heads=4,
                 num_layers=4, feedforward_dim=512, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim

        # Project each feature to model_dim, treating features as a sequence of length input_dim
        # Each feature becomes a token
        self.input_projection = nn.Linear(1, model_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Positional embedding for input_dim + 1 (features + CLS)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim + 1, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.input_dim = input_dim

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) — flat flow feature vector
        Returns:
            (B, model_dim) — [CLS] token embedding
        """
        B = x.shape[0]

        # Reshape to (B, input_dim, 1) then project to (B, input_dim, model_dim)
        x = x.unsqueeze(-1)  # (B, input_dim, 1)
        x = self.input_projection(x)  # (B, input_dim, model_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, model_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, input_dim+1, model_dim)

        # Add positional embedding
        x = x + self.pos_embedding[:, :x.shape[1], :]

        # Transformer
        x = self.transformer(x)  # (B, input_dim+1, model_dim)

        # Take CLS token output
        cls_output = x[:, 0, :]  # (B, model_dim)
        return self.norm(cls_output)
