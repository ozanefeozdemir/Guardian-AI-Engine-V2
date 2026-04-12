#!/usr/bin/env python3
"""Run the full preprocessing pipeline."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import setup_reproducibility
from src.utils.config import load_config
from src.data.preprocess import run_full_preprocessing
from src.data.splits import generate_all_splits


def main():
    parser = argparse.ArgumentParser(description="FlowGuard Preprocessing Pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Config file path")
    parser.add_argument("--skip-splits", action="store_true", help="Skip split generation")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_reproducibility(config)

    print("Running full preprocessing pipeline...")
    all_stats = run_full_preprocessing(args.config)

    if not args.skip_splits:
        print("\nGenerating data splits...")
        generate_all_splits(args.config)

    print("\nPreprocessing complete!")
    print(f"Processed {len(all_stats)} datasets.")


if __name__ == "__main__":
    main()
