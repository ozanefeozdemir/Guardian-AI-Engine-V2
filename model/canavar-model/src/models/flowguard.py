"""
Full FlowGuard model assembling all components.
"""

import torch
import torch.nn as nn
from src.models.transformer_encoder import FlowTransformerEncoder
from src.models.classification_head import ClassificationHead
from src.models.autoencoder import FlowAutoencoder
from src.models.domain_discriminator import DomainDiscriminator


class FlowGuard(nn.Module):
    """
    Complete FlowGuard model.

    Components:
        - encoder: FlowTransformerEncoder (pre-trained via contrastive learning)
        - classifier: ClassificationHead (fine-tuned via federated learning)
        - autoencoder: FlowAutoencoder (optional, for energy score stacking)
        - domain_disc: DomainDiscriminator (optional, for adversarial training)
    """

    def __init__(self, config: dict):
        super().__init__()
        enc_cfg = config.get('model', {}).get('encoder', {})
        self.encoder = FlowTransformerEncoder(
            input_dim=enc_cfg.get('input_dim', 49),
            model_dim=enc_cfg.get('model_dim', 128),
            num_heads=enc_cfg.get('num_heads', 4),
            num_layers=enc_cfg.get('num_layers', 4),
            feedforward_dim=enc_cfg.get('feedforward_dim', 512),
            dropout=enc_cfg.get('dropout', 0.1),
        )

        cls_cfg = config.get('model', {}).get('classification_head', config.get('model', {}))
        num_classes = 2 if cls_cfg.get('output_type', 'binary') == 'binary' else cls_cfg.get('num_classes', 2)
        self.classifier = ClassificationHead(
            input_dim=enc_cfg.get('model_dim', 128),
            hidden_dims=cls_cfg.get('hidden_dims', [64]),
            num_classes=num_classes,
        )

        # Optional components
        self.autoencoder = None
        self.domain_disc = None
        self.use_energy_score = False

    def enable_autoencoder(self, input_dim=49, hidden_dims=None):
        self.autoencoder = FlowAutoencoder(input_dim, hidden_dims)
        self.use_energy_score = True

    def enable_domain_discriminator(self, num_domains=4, hidden_dims=None):
        model_dim = self.encoder.model_dim
        device = next(self.encoder.parameters()).device
        self.domain_disc = DomainDiscriminator(model_dim, hidden_dims, num_domains).to(device)

    def forward(self, x, return_embedding=False):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits

    def forward_with_domain(self, x, lambda_=1.0):
        """Forward pass including domain discrimination."""
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        domain_logits = self.domain_disc(embedding, lambda_) if self.domain_disc else None
        return logits, domain_logits, embedding

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
