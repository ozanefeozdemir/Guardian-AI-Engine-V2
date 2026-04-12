"""
PyTorch Dataset classes for FlowGuard.
Handles loading preprocessed Parquet files into tensors.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler

from src.data.features import LABEL_COLUMNS


class ClassBalancedSampler(Sampler):
    """
    Class-balanced sampler with no 2^24 category limit (avoids torch.multinomial).
    Caps epoch size at max_per_class samples per class so large datasets don't
    produce enormous epochs that starve the GPU.
    """

    def __init__(self, labels: torch.Tensor, max_per_class: int = 100_000):
        self.class_indices = {}
        for cls in torch.unique(labels):
            self.class_indices[cls.item()] = (labels == cls).nonzero(as_tuple=True)[0]
        n_classes = len(self.class_indices)
        self._n_per_class = min(
            max(len(idx) for idx in self.class_indices.values()),
            max_per_class,
        )
        self._total = self._n_per_class * n_classes

    def __iter__(self):
        indices = []
        for cls_idx in self.class_indices.values():
            n = len(cls_idx)
            perm = torch.randperm(n)
            repeats = (self._n_per_class + n - 1) // n  # ceil division
            expanded = cls_idx[perm.repeat(repeats)[:self._n_per_class]]
            indices.append(expanded)
        all_idx = torch.cat(indices)[torch.randperm(self._total)]
        return iter(all_idx.numpy())  # numpy avoids building a 40M-element Python list

    def __len__(self):
        return self._total


class FlowDataset(Dataset):
    """
    PyTorch Dataset for flow records.

    Loads a Parquet file and provides (features, label) pairs.
    Supports binary and multiclass labels.
    """

    def __init__(self, parquet_path: str, label_type: str = "binary",
                 attack_to_idx: dict = None):
        """
        Args:
            parquet_path: Path to a preprocessed .parquet file
            label_type: "binary" (Label col) or "multiclass" (Attack col)
            attack_to_idx: Mapping from attack string to integer index (for multiclass)
        """
        df = pd.read_parquet(parquet_path)

        # Separate features and labels
        label_cols_present = [c for c in LABEL_COLUMNS if c in df.columns]
        feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]

        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)

        if label_type == "binary" and "Label" in df.columns:
            self.labels = torch.tensor(df["Label"].values, dtype=torch.long)
        elif label_type == "multiclass" and "Attack" in df.columns:
            if attack_to_idx is None:
                unique_attacks = sorted(df["Attack"].unique())
                attack_to_idx = {a: i for i, a in enumerate(unique_attacks)}
            self.labels = torch.tensor(
                df["Attack"].map(attack_to_idx).fillna(0).astype(int).values,
                dtype=torch.long
            )
            self.attack_to_idx = attack_to_idx
        else:
            # No labels (unlabeled dataset for SSL)
            self.labels = None

        self.label_type = label_type
        self.feature_names = feature_cols

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class UnlabeledFlowDataset(Dataset):
    """Dataset for self-supervised pre-training (no labels)."""

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        # Drop any label columns if present
        feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def create_dataloader(
    parquet_path: str,
    batch_size: int = 1024,
    label_type: str = "binary",
    shuffle: bool = True,
    balanced: bool = False,
    num_workers: int = 0,
    attack_to_idx: dict = None,
) -> DataLoader:
    """
    Create a DataLoader from a Parquet file.

    Args:
        balanced: If True, use ClassBalancedSampler for class-balanced batches
    """
    dataset = FlowDataset(parquet_path, label_type=label_type, attack_to_idx=attack_to_idx)

    sampler = None
    if balanced and dataset.labels is not None:
        sampler = ClassBalancedSampler(dataset.labels)
        shuffle = False  # Sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def create_unlabeled_dataloader(
    parquet_path: str,
    batch_size: int = 4096,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for unlabeled SSL data."""
    dataset = UnlabeledFlowDataset(parquet_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for contrastive learning
    )
