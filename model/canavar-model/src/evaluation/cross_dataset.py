"""Cross-dataset evaluation harness."""

import os
import json
import torch
from datetime import datetime

from src.utils.config import get_device
from src.evaluation.protocols import (
    evaluate_protocol_a, evaluate_protocol_b,
    evaluate_protocol_c, generate_comparison_table
)


def run_full_evaluation(model, config: dict, phase_name: str = "unknown",
                        model_factory=None, adaptation_fn=None) -> dict:
    """
    Run all three evaluation protocols and save results.

    Args:
        model: Trained model for Protocol A and B
        config: Config dict
        phase_name: Name of the phase (for result files)
        model_factory: For Protocol C (creates fresh models)
        adaptation_fn: For Protocol C (adapts model with few shots)
    """
    device = get_device(config)
    model.to(device)

    results = {}

    print("=" * 60)
    print(f"EVALUATION: {phase_name}")
    print("=" * 60)

    print("\n--- Protocol A: Within-dataset ---")
    results['protocol_a'] = evaluate_protocol_a(model, config, device)

    print("\n--- Protocol B: Cross-dataset ---")
    results['protocol_b'] = evaluate_protocol_b(model, config, device)

    if model_factory and adaptation_fn:
        print("\n--- Protocol C: Few-shot ---")
        results['protocol_c'] = evaluate_protocol_c(model_factory, config, adaptation_fn, device)

    # Save results
    results_dir = "results/metrics"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{phase_name}_{timestamp}.json")

    # Convert ProtocolResult objects to dicts for JSON
    serializable = {}
    for protocol, result_list in results.items():
        serializable[protocol] = [
            {
                'protocol': r.protocol,
                'train_dataset': r.train_dataset,
                'test_dataset': r.test_dataset,
                'shots': r.shots,
                'accuracy': r.accuracy,
                'f1_macro': r.f1_macro,
                'auroc': r.auroc,
                'fpr': r.fpr,
                'f1_per_class': r.f1_per_class,
            }
            for r in result_list
        ]

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {results_file}")
    return results
