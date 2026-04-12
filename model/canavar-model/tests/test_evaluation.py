"""Tests for evaluation metrics."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_all_metrics, format_metrics


class TestMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        metrics = compute_all_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_macro'] == 1.0
        assert metrics['fpr'] == 0.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        metrics = compute_all_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 0.0
        assert metrics['fpr'] == 1.0

    def test_auroc_with_probs(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert 'auroc' in metrics
        assert metrics['auroc'] == 1.0

    def test_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])

        metrics = compute_all_metrics(y_true, y_pred)
        cm = metrics['confusion_matrix']
        assert len(cm) == 2
        assert cm[0][0] == 2  # TN
        assert cm[0][1] == 1  # FP
        assert cm[1][0] == 0  # FN
        assert cm[1][1] == 2  # TP

    def test_format_metrics(self):
        metrics = {'accuracy': 0.95, 'f1_macro': 0.93, 'fpr': 0.02}
        output = format_metrics(metrics)
        assert 'Accuracy' in output
        assert '0.95' in output


class TestDriftMonitor:

    def test_no_drift(self):
        from src.evaluation.drift_monitor import DriftMonitor

        np.random.seed(42)
        ref_data = np.random.randn(1000, 10)
        new_data = np.random.randn(500, 10)

        monitor = DriftMonitor(ref_data, threshold=0.5)
        result = monitor.check_drift(new_data)

        assert 'drifted' in result
        assert 'mean_kl' in result
        # Same distribution should have low KL
        assert result['mean_kl'] < 0.5

    def test_drift_detected(self):
        from src.evaluation.drift_monitor import DriftMonitor

        np.random.seed(42)
        ref_data = np.random.randn(1000, 10)
        # Shifted distribution
        new_data = np.random.randn(500, 10) + 5

        monitor = DriftMonitor(ref_data, threshold=0.1)
        result = monitor.check_drift(new_data)

        assert result['drifted'] is True
        assert result['mean_kl'] > 0.1
