"""
Generate reproducible train/val/test splits for all three protocols.
CRITICAL: Never shuffle then split. Maintain temporal ordering where possible.
"""

import os
import pandas as pd
import numpy as np
from src.utils.config import load_config


def generate_protocol_a_splits(config: dict) -> None:
    """
    Protocol A: Within-dataset evaluation.
    For each of the 4 datasets:
      - 80% train, 10% validation, 10% test
      - Split by index order (proxy for temporal)
      - Save as data/splits/protocol_a/{dataset_name}_{split}.parquet
    """
    processed_dir = config['data']['processed_dir']
    splits_dir = os.path.join(config['data']['splits_dir'], 'protocol_a')
    os.makedirs(splits_dir, exist_ok=True)

    for ds_info in config['data']['datasets']:
        name = ds_info['name']
        parquet_path = os.path.join(processed_dir, f"{name}.parquet")

        if not os.path.exists(parquet_path):
            print(f"Skipping {name}: {parquet_path} not found. Run preprocessing first.")
            continue

        df = pd.read_parquet(parquet_path)
        n = len(df)

        # 80/10/10 split by index order (temporal proxy)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        train_df.to_parquet(os.path.join(splits_dir, f"{name}_train.parquet"), index=False)
        val_df.to_parquet(os.path.join(splits_dir, f"{name}_val.parquet"), index=False)
        test_df.to_parquet(os.path.join(splits_dir, f"{name}_test.parquet"), index=False)

        print(f"Protocol A - {name}: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")


def generate_protocol_b_splits(config: dict) -> None:
    """
    Protocol B: Cross-dataset evaluation (leave-one-dataset-out).
    For each dataset as held-out test:
      - Train: remaining 3 datasets combined
      - Validation: 10% of training data (stratified by Label)
      - Test: entire held-out dataset
      - Save as data/splits/protocol_b/holdout_{dataset_name}/{split}.parquet
    """
    processed_dir = config['data']['processed_dir']
    splits_base = os.path.join(config['data']['splits_dir'], 'protocol_b')
    seed = config['project']['seed']

    dataset_names = [ds['name'] for ds in config['data']['datasets']]
    datasets = {}

    for name in dataset_names:
        path = os.path.join(processed_dir, f"{name}.parquet")
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found.")
            continue
        datasets[name] = pd.read_parquet(path)

    for holdout_name in dataset_names:
        if holdout_name not in datasets:
            continue

        split_dir = os.path.join(splits_base, f"holdout_{holdout_name}")
        os.makedirs(split_dir, exist_ok=True)

        # Combine training datasets
        train_dfs = [datasets[n] for n in dataset_names if n != holdout_name and n in datasets]
        combined_train = pd.concat(train_dfs, ignore_index=True)

        # Stratified 90/10 train/val split
        from sklearn.model_selection import train_test_split
        label_col = config['data']['features']['label_binary']

        train_df, val_df = train_test_split(
            combined_train, test_size=0.1, random_state=seed,
            stratify=combined_train[label_col]
        )

        test_df = datasets[holdout_name]

        train_df.to_parquet(os.path.join(split_dir, "train.parquet"), index=False)
        val_df.to_parquet(os.path.join(split_dir, "val.parquet"), index=False)
        test_df.to_parquet(os.path.join(split_dir, "test.parquet"), index=False)

        print(f"Protocol B - holdout {holdout_name}: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")


def generate_protocol_c_splits(config: dict) -> None:
    """
    Protocol C: Few-shot adaptation.
    Same as Protocol B, but additionally sample small labeled subsets from held-out for adaptation.
    """
    processed_dir = config['data']['processed_dir']
    splits_base = os.path.join(config['data']['splits_dir'], 'protocol_c')
    seed = config['project']['seed']
    label_col = config['data']['features']['label_binary']

    shot_counts = [5, 10, 20, 50]

    dataset_names = [ds['name'] for ds in config['data']['datasets']]
    datasets = {}
    for name in dataset_names:
        path = os.path.join(processed_dir, f"{name}.parquet")
        if os.path.exists(path):
            datasets[name] = pd.read_parquet(path)

    for holdout_name in dataset_names:
        if holdout_name not in datasets:
            continue

        split_dir = os.path.join(splits_base, f"holdout_{holdout_name}")
        os.makedirs(split_dir, exist_ok=True)

        # Same train/val as Protocol B
        train_dfs = [datasets[n] for n in dataset_names if n != holdout_name and n in datasets]
        combined_train = pd.concat(train_dfs, ignore_index=True)

        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            combined_train, test_size=0.1, random_state=seed,
            stratify=combined_train[label_col]
        )

        train_df.to_parquet(os.path.join(split_dir, "train.parquet"), index=False)
        val_df.to_parquet(os.path.join(split_dir, "val.parquet"), index=False)

        holdout_df = datasets[holdout_name]

        for shots in shot_counts:
            rng = np.random.RandomState(seed)

            # Sample N per class
            adapt_indices = []
            for label_val in holdout_df[label_col].unique():
                class_indices = holdout_df[holdout_df[label_col] == label_val].index.tolist()
                n_sample = min(shots, len(class_indices))
                sampled = rng.choice(class_indices, size=n_sample, replace=False)
                adapt_indices.extend(sampled)

            adapt_df = holdout_df.loc[adapt_indices]
            test_df = holdout_df.drop(index=adapt_indices)

            adapt_df.to_parquet(os.path.join(split_dir, f"adapt_{shots}shot.parquet"), index=False)
            test_df.to_parquet(os.path.join(split_dir, f"test_{shots}shot.parquet"), index=False)

            print(f"Protocol C - holdout {holdout_name}, {shots}-shot: adapt={len(adapt_df)}, test={len(test_df):,}")


def generate_all_splits(config_path: str = "configs/base.yaml") -> None:
    """Generate all protocol splits."""
    config = load_config(config_path)
    print("=" * 60)
    print("Generating Protocol A splits (within-dataset)...")
    print("=" * 60)
    generate_protocol_a_splits(config)

    print("\n" + "=" * 60)
    print("Generating Protocol B splits (cross-dataset)...")
    print("=" * 60)
    generate_protocol_b_splits(config)

    print("\n" + "=" * 60)
    print("Generating Protocol C splits (few-shot)...")
    print("=" * 60)
    generate_protocol_c_splits(config)

    print("\nAll splits generated successfully.")
