"""SHAP analysis and feature importance for FlowGuard."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_shap_values(model, data_loader, feature_names, device,
                        num_samples=500, background_samples=100):
    """
    Compute SHAP values for model predictions.

    Uses KernelSHAP for model-agnostic explanation.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

    model.eval()

    # Collect background and explanation samples
    all_x = []
    count = 0
    for batch in data_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        all_x.append(x)
        count += x.shape[0]
        if count >= num_samples + background_samples:
            break

    all_x = torch.cat(all_x)[:num_samples + background_samples]

    background = all_x[:background_samples].numpy()
    explain_data = all_x[background_samples:background_samples + num_samples].numpy()

    # Wrap model for SHAP
    def model_predict(x_np):
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
            logits = model(x_t)
            return torch.softmax(logits, dim=1).cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(explain_data, nsamples=100)

    return shap_values, explain_data, feature_names


def plot_shap_summary(shap_values, data, feature_names, save_path=None):
    """Generate and save SHAP summary plot."""
    import shap

    # For binary classification, use the attack class SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]
    else:
        sv = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, data, feature_names=feature_names, show=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP plot saved to {save_path}")

    plt.close()


def get_feature_importance(shap_values, feature_names):
    """Get mean absolute SHAP values as feature importance."""
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]
    else:
        sv = shap_values

    importance = np.abs(sv).mean(axis=0)
    ranked = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    return ranked
