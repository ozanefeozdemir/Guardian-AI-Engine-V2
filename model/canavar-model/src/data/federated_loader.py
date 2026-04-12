"""
Per-silo DataLoaders for federated learning.
Each dataset acts as a separate client/silo.
"""

import os
from torch.utils.data import DataLoader

from src.data.dataset import FlowDataset, create_dataloader
from src.utils.config import load_config


def create_federated_loaders(
    config: dict,
    split_protocol: str = "protocol_a",
) -> dict:
    """
    Create per-silo train and validation DataLoaders.

    Returns:
        {
            "unsw": {"train": DataLoader, "val": DataLoader, "test": DataLoader},
            "bot": {"train": DataLoader, "val": DataLoader, "test": DataLoader},
            ...
        }
    """
    splits_dir = os.path.join(config['data']['splits_dir'], split_protocol)
    batch_size = config.get('federated', {}).get('client', {}).get('batch_size', 1024)
    balanced = config.get('federated', {}).get('client', {}).get('balanced_sampling', True)

    num_workers = config.get('federated', {}).get('client', {}).get('num_workers', 2)

    loaders = {}
    for ds_info in config['data']['datasets']:
        name = ds_info['name']
        silo_loaders = {}

        for split in ['train', 'val', 'test']:
            path = os.path.join(splits_dir, f"{name}_{split}.parquet")
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping {name}/{split}")
                continue

            silo_loaders[split] = create_dataloader(
                parquet_path=path,
                batch_size=batch_size,
                label_type="binary",
                shuffle=(split == "train"),
                balanced=(balanced and split == "train"),
                num_workers=num_workers,
            )

        if silo_loaders:
            loaders[name] = silo_loaders

    return loaders
