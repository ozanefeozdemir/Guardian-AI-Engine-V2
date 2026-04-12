"""UMAP/t-SNE embedding visualization."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=5000):
    """Extract embeddings from model encoder."""
    model.eval()
    embeddings = []
    labels = []
    count = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0].to(device), batch[1]
            labels.append(y.numpy())
        else:
            x = batch.to(device)

        # Try to get embeddings from encoder
        if hasattr(model, 'encoder'):
            emb = model.encoder(x)
        elif hasattr(model, 'get_features'):
            emb = model.get_features(x)
        else:
            emb = model(x)

        embeddings.append(emb.cpu().numpy())
        count += x.shape[0]
        if count >= max_samples:
            break

    embeddings = np.concatenate(embeddings)[:max_samples]
    if labels:
        labels = np.concatenate(labels)[:max_samples]
    else:
        labels = None

    return embeddings, labels


def plot_umap(embeddings, labels=None, label_names=None, title="UMAP Embeddings",
              save_path=None, dataset_ids=None):
    """
    Generate UMAP visualization of embeddings.

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) integer labels (optional)
        label_names: Mapping from label int to name
        title: Plot title
        save_path: Path to save figure
        dataset_ids: (N,) dataset identifier for multi-dataset visualization
    """
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Run: pip install umap-learn")
        return

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names[label] if label_names and label in label_names else str(label)
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name,
                      alpha=0.5, s=5)
        ax.legend(markerscale=3, fontsize=8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=5)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"UMAP plot saved to {save_path}")

    plt.close(fig)


def plot_tsne(embeddings, labels=None, label_names=None, title="t-SNE Embeddings",
              save_path=None):
    """Generate t-SNE visualization."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names[label] if label_names and label in label_names else str(label)
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name,
                      alpha=0.5, s=5)
        ax.legend(markerscale=3, fontsize=8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=5)

    ax.set_title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close(fig)
