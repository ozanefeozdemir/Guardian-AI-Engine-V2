"""Concept drift detection using KL divergence monitoring."""

import numpy as np
from scipy import stats


class DriftMonitor:
    """
    Monitors feature distribution drift using KL divergence.
    Compares current batch statistics against reference distribution.
    """

    def __init__(self, reference_data: np.ndarray, n_bins: int = 50,
                 threshold: float = 0.1):
        """
        Args:
            reference_data: (N, D) reference feature matrix (training data)
            n_bins: Number of histogram bins for KL computation
            threshold: KL divergence threshold for drift alarm
        """
        self.n_features = reference_data.shape[1]
        self.n_bins = n_bins
        self.threshold = threshold

        # Compute reference histograms per feature
        self.ref_histograms = []
        self.bin_edges = []

        for i in range(self.n_features):
            hist, edges = np.histogram(reference_data[:, i], bins=n_bins, density=True)
            hist = hist + 1e-10  # Avoid zero bins
            hist = hist / hist.sum()
            self.ref_histograms.append(hist)
            self.bin_edges.append(edges)

    def check_drift(self, new_data: np.ndarray) -> dict:
        """
        Check if new data distribution has drifted from reference.

        Returns:
            {
                'drifted': bool,
                'kl_per_feature': list[float],
                'mean_kl': float,
                'max_kl': float,
                'drifted_features': list[int],
            }
        """
        kl_values = []
        drifted_features = []

        for i in range(self.n_features):
            new_hist, _ = np.histogram(new_data[:, i], bins=self.bin_edges[i], density=True)
            new_hist = new_hist + 1e-10
            new_hist = new_hist / new_hist.sum()

            kl = stats.entropy(self.ref_histograms[i], new_hist)
            kl_values.append(float(kl))

            if kl > self.threshold:
                drifted_features.append(i)

        return {
            'drifted': len(drifted_features) > 0,
            'kl_per_feature': kl_values,
            'mean_kl': float(np.mean(kl_values)),
            'max_kl': float(np.max(kl_values)),
            'drifted_features': drifted_features,
        }
