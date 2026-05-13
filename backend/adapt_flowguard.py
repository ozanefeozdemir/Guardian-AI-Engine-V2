"""Thin CLI wrapper around canavar-model's Phase 4 (Few-shot) adaptation.

Mirrors model/canavar-model/scripts/adapt_fewshot.py. We don't duplicate the
adaptation pipeline here — we just expose it from the backend tree so the
engine and adaptation share one entry-point convention.

Run:
    python backend/adapt_flowguard.py
    python backend/adapt_flowguard.py --config <path>
"""

import os
import sys
import json
import argparse

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
CANAVAR_SRC = os.path.join(PROJECT_ROOT, "model", "canavar-model")
DEFAULT_CONFIG = os.path.join(CANAVAR_SRC, "configs", "phase4_fewshot.yaml")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "metrics", "phase4_fewshot.json")

if CANAVAR_SRC not in sys.path:
    sys.path.insert(0, CANAVAR_SRC)


def main():
    parser = argparse.ArgumentParser(
        description="FlowGuard few-shot adaptation (canavar-model Phase 4)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.utils.reproducibility import setup_reproducibility
    from src.training.fewshot import run_fewshot_evaluation

    config = load_config(args.config)
    setup_reproducibility(config)

    results = run_fewshot_evaluation(args.config)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFew-shot adaptation complete. Results: {RESULTS_PATH}")
    for ds, shots_results in results.items():
        for shots, metrics in shots_results.items():
            print(f"  {ds} {shots}-shot: F1={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
