"""
Data augmentations for self-supervised contrastive learning on flow records.

These augmentations create two different "views" of each flow record.
The contrastive loss learns to map the two views close together
while pushing views of different flows apart.

DESIGN PRINCIPLE: Augmentations simulate natural variation across
different network collectors and environments, NOT create unrealistic data.
"""

import torch
from src.data.features import get_feature_names, TIMING_FEATURES, IAT_FEATURES, SWAP_PAIRS


# Precompute feature indices for swap pairs
_FEATURE_NAMES = get_feature_names()

# Forward/backward swap pairs (precomputed indices from features.SWAP_PAIRS)
_SWAP_PAIRS = []
for fwd, bwd in SWAP_PAIRS:
    if fwd in _FEATURE_NAMES and bwd in _FEATURE_NAMES:
        _SWAP_PAIRS.append((_FEATURE_NAMES.index(fwd), _FEATURE_NAMES.index(bwd)))

# Timing feature indices (for temporal jitter)
_TIMING_INDICES = [
    _FEATURE_NAMES.index(f)
    for f in TIMING_FEATURES + IAT_FEATURES
    if f in _FEATURE_NAMES
]


class FlowAugmentor:
    """
    Produces augmented views of flow feature tensors.

    Usage:
        aug = FlowAugmentor(config)
        view1 = aug(batch)  # First augmented view
        view2 = aug(batch)  # Second augmented view (different random augmentation)
    """

    def __init__(self, config: dict):
        """
        Config fields used:
          feature_masking.enabled, feature_masking.mask_ratio
          gaussian_noise.enabled, gaussian_noise.std
          direction_swap.enabled, direction_swap.probability
          temporal_jitter.enabled, temporal_jitter.jitter_ratio
        """
        fm = config.get('feature_masking', {})
        self.mask_enabled = fm.get('enabled', True)
        self.mask_ratio = fm.get('mask_ratio', 0.2)

        gn = config.get('gaussian_noise', {})
        self.noise_enabled = gn.get('enabled', True)
        self.noise_std = gn.get('std', 0.05)

        ds = config.get('direction_swap', {})
        self.swap_enabled = ds.get('enabled', True)
        self.swap_prob = ds.get('probability', 0.1)

        tj = config.get('temporal_jitter', {})
        self.jitter_enabled = tj.get('enabled', True)
        self.jitter_ratio = tj.get('jitter_ratio', 0.05)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to a batch of flow features.

        Args:
            x: Tensor of shape (batch_size, num_features)
        Returns:
            Augmented tensor of same shape
        """
        x = x.clone()

        if self.mask_enabled:
            x = self._feature_masking(x)
        if self.noise_enabled:
            x = self._gaussian_noise(x)
        if self.swap_enabled:
            x = self._direction_swap(x)
        if self.jitter_enabled:
            x = self._temporal_jitter(x)

        return x

    def _feature_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out mask_ratio fraction of features per sample."""
        mask = torch.rand_like(x) > self.mask_ratio
        return x * mask.float()

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise N(0, std) to all features."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def _direction_swap(self, x: torch.Tensor) -> torch.Tensor:
        """
        With given probability, swap forward<->backward feature pairs.
        E.g., swap IN_BYTES <-> OUT_BYTES, FWD_IAT_MEAN <-> BWD_IAT_MEAN.
        """
        if not _SWAP_PAIRS:
            return x

        # Per-sample swap decision
        swap_mask = torch.rand(x.shape[0], device=x.device) < self.swap_prob

        if swap_mask.any():
            for idx_a, idx_b in _SWAP_PAIRS:
                temp = x[swap_mask, idx_a].clone()
                x[swap_mask, idx_a] = x[swap_mask, idx_b]
                x[swap_mask, idx_b] = temp

        return x

    def _temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add +/-jitter_ratio relative perturbation to timing features only."""
        if not _TIMING_INDICES:
            return x

        jitter = 1.0 + (torch.rand(x.shape[0], len(_TIMING_INDICES), device=x.device) * 2 - 1) * self.jitter_ratio
        x[:, _TIMING_INDICES] = x[:, _TIMING_INDICES] * jitter

        return x
