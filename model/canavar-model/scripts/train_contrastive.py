#!/usr/bin/env python3
"""Run Phase 2: Self-supervised contrastive pre-training."""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import setup_reproducibility
from src.utils.config import load_config
from src.training.contrastive import train_contrastive


def main():
    parser = argparse.ArgumentParser(description="Contrastive Pre-training (Phase 2)")
    parser.add_argument("--config", default="configs/phase2_contrastive.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_reproducibility(config)

    history = train_contrastive(args.config)
    print(f"\nPhase 2 complete! Final loss: {history['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
