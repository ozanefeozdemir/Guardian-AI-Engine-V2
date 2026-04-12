"""All metric computations for FlowGuard evaluation."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_fscore_support
)


def compute_all_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Compute comprehensive metrics.

    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted labels (numpy array)
        y_prob: Predicted probabilities (numpy array, optional for AUROC)
        class_names: List of class names for per-class reporting

    Returns:
        dict with all metrics
    """
    results = {}

    # Basic metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class F1
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)

    if class_names and len(class_names) == len(unique_labels):
        results['f1_per_class'] = {name: float(f1) for name, f1 in zip(class_names, per_class_f1)}
    else:
        results['f1_per_class'] = {str(label): float(f1) for label, f1 in zip(unique_labels, per_class_f1)}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    results['confusion_matrix'] = cm.tolist()

    # False Positive Rate (for binary classification)
    if len(unique_labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        results['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        results['tpr'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        results['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        results['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # AUROC (requires probability scores)
    if y_prob is not None:
        try:
            if len(unique_labels) == 2:
                # Binary: use probability of positive class
                if y_prob.ndim == 2:
                    results['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    results['auroc'] = roc_auc_score(y_true, y_prob)
            else:
                # Multiclass: one-vs-rest
                results['auroc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            results['auroc'] = 0.0

    return results


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=2):
    """
    Evaluate a model on a dataloader.

    Returns:
        dict with all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0].to(device), batch[1]
        else:
            continue

        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(y)
        all_probs.append(probs)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    class_names = ['Benign', 'Attack'] if num_classes == 2 else None
    return compute_all_metrics(all_labels, all_preds, all_probs, class_names)


def format_metrics(metrics: dict) -> str:
    """Format metrics dict as a human-readable string."""
    lines = []
    lines.append(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    lines.append(f"F1 Macro:  {metrics.get('f1_macro', 0):.4f}")

    if 'auroc' in metrics:
        lines.append(f"AUROC:     {metrics['auroc']:.4f}")
    if 'fpr' in metrics:
        lines.append(f"FPR:       {metrics['fpr']:.4f}")

    if 'f1_per_class' in metrics:
        lines.append("Per-class F1:")
        for cls, f1 in metrics['f1_per_class'].items():
            lines.append(f"  {cls}: {f1:.4f}")

    return '\n'.join(lines)
