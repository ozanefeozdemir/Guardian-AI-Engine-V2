"""Standardized evaluation protocols A, B, C."""

import os
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
from src.utils.config import load_config, get_device
from src.data.dataset import create_dataloader
from src.evaluation.metrics import evaluate_model, format_metrics


@dataclass
class ProtocolResult:
    """Result container for a single protocol evaluation."""
    protocol: str
    train_dataset: str
    test_dataset: str
    shots: int = 0
    accuracy: float = 0.0
    f1_per_class: dict = field(default_factory=dict)
    f1_macro: float = 0.0
    auroc: float = 0.0
    fpr: float = 0.0
    confusion_matrix: list = field(default_factory=list)


def evaluate_protocol_a(model, config: dict, device=None) -> list:
    """
    Within-dataset evaluation.
    For each dataset: evaluate model on its test split.
    """
    if device is None:
        device = get_device(config)

    results = []
    splits_dir = os.path.join(config['data']['splits_dir'], 'protocol_a')

    for ds_info in config['data']['datasets']:
        name = ds_info['name']
        test_path = os.path.join(splits_dir, f"{name}_test.parquet")

        if not os.path.exists(test_path):
            print(f"Skipping {name}: test split not found")
            continue

        test_loader = create_dataloader(test_path, batch_size=1024, shuffle=False)
        metrics = evaluate_model(model, test_loader, device)

        result = ProtocolResult(
            protocol="a",
            train_dataset=name,
            test_dataset=name,
            accuracy=metrics['accuracy'],
            f1_per_class=metrics.get('f1_per_class', {}),
            f1_macro=metrics['f1_macro'],
            auroc=metrics.get('auroc', 0),
            fpr=metrics.get('fpr', 0),
            confusion_matrix=metrics.get('confusion_matrix', []),
        )
        results.append(result)
        print(f"Protocol A - {name}:")
        print(format_metrics(metrics))
        print()

    return results


def evaluate_protocol_b(model, config: dict, device=None) -> list:
    """
    Cross-dataset evaluation (leave-one-out).
    Evaluate model on each held-out dataset's test split.
    """
    if device is None:
        device = get_device(config)

    results = []

    for ds_info in config['data']['datasets']:
        name = ds_info['name']
        test_path = os.path.join(config['data']['splits_dir'],
                                  'protocol_b', f'holdout_{name}', 'test.parquet')

        if not os.path.exists(test_path):
            print(f"Skipping holdout {name}: test split not found")
            continue

        test_loader = create_dataloader(test_path, batch_size=1024, shuffle=False)
        metrics = evaluate_model(model, test_loader, device)

        train_names = [d['name'] for d in config['data']['datasets'] if d['name'] != name]

        result = ProtocolResult(
            protocol="b",
            train_dataset="+".join(train_names),
            test_dataset=name,
            accuracy=metrics['accuracy'],
            f1_per_class=metrics.get('f1_per_class', {}),
            f1_macro=metrics['f1_macro'],
            auroc=metrics.get('auroc', 0),
            fpr=metrics.get('fpr', 0),
            confusion_matrix=metrics.get('confusion_matrix', []),
        )
        results.append(result)
        print(f"Protocol B - holdout {name}:")
        print(format_metrics(metrics))
        print()

    return results


def evaluate_protocol_c(model_factory, config: dict, adaptation_fn=None,
                         device=None) -> list:
    """
    Few-shot adaptation evaluation.

    Args:
        model_factory: Callable that returns a fresh model (for each adaptation)
        config: Config dict
        adaptation_fn: Function(model, adapt_loader, config, device) -> adapted model
        device: Torch device
    """
    if device is None:
        device = get_device(config)

    results = []
    shot_counts = config.get('fewshot', {}).get('shots', [5, 10, 20, 50])

    for ds_info in config['data']['datasets']:
        name = ds_info['name']

        for shots in shot_counts:
            split_dir = os.path.join(config['data']['splits_dir'],
                                     'protocol_c', f'holdout_{name}')
            adapt_path = os.path.join(split_dir, f"adapt_{shots}shot.parquet")
            test_path = os.path.join(split_dir, f"test_{shots}shot.parquet")

            if not os.path.exists(adapt_path) or not os.path.exists(test_path):
                continue

            # Fresh model for each evaluation
            model = model_factory()
            model.to(device)

            if adaptation_fn:
                adapt_loader = create_dataloader(adapt_path, batch_size=min(shots*2, 64), shuffle=True)
                model = adaptation_fn(model, adapt_loader, config, device)

            test_loader = create_dataloader(test_path, batch_size=1024, shuffle=False)
            metrics = evaluate_model(model, test_loader, device)

            result = ProtocolResult(
                protocol="c",
                train_dataset="all_except_" + name,
                test_dataset=name,
                shots=shots,
                accuracy=metrics['accuracy'],
                f1_per_class=metrics.get('f1_per_class', {}),
                f1_macro=metrics['f1_macro'],
                auroc=metrics.get('auroc', 0),
                fpr=metrics.get('fpr', 0),
                confusion_matrix=metrics.get('confusion_matrix', []),
            )
            results.append(result)

    return results


def generate_comparison_table(results: dict) -> str:
    """
    Generate formatted markdown comparison table across phases.

    Args:
        results: {"phase1": [ProtocolResult, ...], "phase2": [...], ...}
    """
    lines = []
    lines.append("| Phase | Protocol | Test Dataset | Shots | Accuracy | F1 Macro | AUROC | FPR |")
    lines.append("|-------|----------|-------------|-------|----------|----------|-------|-----|")

    for phase_name, phase_results in results.items():
        for r in phase_results:
            lines.append(
                f"| {phase_name} | {r.protocol.upper()} | {r.test_dataset} | "
                f"{r.shots if r.shots > 0 else '-'} | "
                f"{r.accuracy:.4f} | {r.f1_macro:.4f} | "
                f"{r.auroc:.4f} | {r.fpr:.4f} |"
            )

    return '\n'.join(lines)
