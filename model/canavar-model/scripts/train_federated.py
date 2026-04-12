#!/usr/bin/env python3
"""Run Phase 3: Federated Learning."""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import setup_reproducibility
from src.utils.config import load_config
from src.training.federated import run_federated_simulation


def main():
    parser = argparse.ArgumentParser(description="Federated Learning (Phase 3)")
    parser.add_argument("--config", default="configs/phase3_federated.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_reproducibility(config)

    history = run_federated_simulation(args.config)
    print(f"\nPhase 3 complete! {len(history['rounds'])} rounds trained.")


if __name__ == "__main__":
    main()
