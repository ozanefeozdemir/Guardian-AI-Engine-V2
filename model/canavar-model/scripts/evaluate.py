#!/usr/bin/env python3
"""Run evaluation on a trained model."""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_config, get_device
from src.utils.reproducibility import setup_reproducibility
from src.models.flowguard import FlowGuard
from src.evaluation.cross_dataset import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate FlowGuard Model")
    parser.add_argument("--protocol", default="b", choices=["a", "b", "c", "all"])
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--phase-name", default="evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_reproducibility(config)
    device = get_device(config)

    # Load model
    model = FlowGuard(config)
    state = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)

    results = run_full_evaluation(model, config, phase_name=args.phase_name)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
