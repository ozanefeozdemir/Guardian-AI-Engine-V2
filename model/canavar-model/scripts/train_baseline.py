#!/usr/bin/env python3
"""Train Phase 1 MLP baseline."""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, get_device
from src.utils.reproducibility import setup_reproducibility
from src.models.mlp_baseline import MLPBaseline
from src.data.dataset import create_dataloader
from src.training.supervised import SupervisedTrainer
from src.evaluation.protocols import evaluate_protocol_a, evaluate_protocol_b
from src.data.features import get_input_dim


def main():
    parser = argparse.ArgumentParser(description="Train MLP Baseline (Phase 1)")
    parser.add_argument("--config", default="configs/phase1_baseline.yaml")
    parser.add_argument("--dataset", default=None, help="Train on specific dataset only")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_reproducibility(config)
    device = get_device(config)

    model_cfg = config.get('model', {})
    input_dim = model_cfg.get('input_dim', get_input_dim())
    hidden_dims = model_cfg.get('hidden_dims', [256, 128, 64])
    num_classes = 2 if model_cfg.get('output_type', 'binary') == 'binary' else 12
    dropout = model_cfg.get('dropout', 0.3)

    # Train on each dataset (Protocol A)
    datasets = config['data']['datasets']
    if args.dataset:
        datasets = [d for d in datasets if d['name'] == args.dataset]

    for ds_info in datasets:
        name = ds_info['name']
        print(f"\n{'='*60}")
        print(f"Training MLP on: {name}")
        print(f"{'='*60}")

        train_path = f"data/splits/protocol_a/{name}_train.parquet"
        val_path = f"data/splits/protocol_a/{name}_val.parquet"

        if not os.path.exists(train_path):
            print(f"  Skipping {name}: splits not found. Run preprocessing first.")
            continue

        train_loader = create_dataloader(
            train_path,
            batch_size=config.get('training', {}).get('batch_size', 1024),
            balanced=True,
        )
        val_loader = create_dataloader(val_path, batch_size=1024, shuffle=False)

        model = MLPBaseline(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

        trainer = SupervisedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_dir=f"checkpoints/phase1/{name}/",
        )

        history = trainer.train()

        # Load best model and evaluate
        trainer.load_checkpoint()
        print(f"\n--- Protocol A Results for {name} ---")
        evaluate_protocol_a(model, config, device)

    print("\nPhase 1 training complete!")


if __name__ == "__main__":
    main()
