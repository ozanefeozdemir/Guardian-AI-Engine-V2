#!/usr/bin/env python3
"""Run Phase 4: Few-shot Adaptation."""

import sys
import os
import argparse
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import setup_reproducibility
from src.utils.config import load_config
from src.training.fewshot import run_fewshot_evaluation


def main():
    parser = argparse.ArgumentParser(description="Few-shot Adaptation (Phase 4)")
    parser.add_argument("--config", default="configs/phase4_fewshot.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_reproducibility(config)

    results = run_fewshot_evaluation(args.config)

    # Save results
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/phase4_fewshot.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nPhase 4 complete!")
    for ds, shots_results in results.items():
        for shots, metrics in shots_results.items():
            print(f"  {ds} {shots}-shot: F1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
