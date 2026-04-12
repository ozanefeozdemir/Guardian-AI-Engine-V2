"""Few-shot adaptation logic (Phase 4)."""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device
from src.models.transformer_encoder import FlowTransformerEncoder
from src.models.classification_head import ClassificationHead
from src.data.dataset import FlowDataset, create_dataloader


def head_finetune(encoder, adapt_loader, test_loader, config, device):
    """
    Few-shot adaptation by fine-tuning only the classification head.
    Encoder is frozen.
    """
    fs_cfg = config.get('fewshot', {}).get('head_finetune', {})

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # New classification head
    model_dim = encoder.model_dim
    head = ClassificationHead(input_dim=model_dim, hidden_dims=[64], num_classes=2).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=fs_cfg.get('lr', 0.001))
    criterion = nn.CrossEntropyLoss()
    patience = fs_cfg.get('early_stopping_patience', 10)
    max_epochs = fs_cfg.get('epochs', 50)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        head.train()
        epoch_loss = 0
        n = 0

        for x, y in adapt_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = encoder(x)
            logits = head(features)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)

        avg_loss = epoch_loss / max(n, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    # Evaluate
    head.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            features = encoder(x)
            logits = head(features)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import f1_score, accuracy_score
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
    }


def run_fewshot_evaluation(config_path: str = "configs/phase4_fewshot.yaml") -> dict:
    """Run few-shot adaptation evaluation across all held-out datasets and shot counts."""
    config = load_config(config_path)
    device = get_device(config)

    # Load encoder
    encoder_path = "checkpoints/phase2/best_encoder.pt"
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Missing encoder: {encoder_path}. Run Phase 2 first.")

    enc_cfg = config.get('model', {}).get('encoder', {})
    encoder = FlowTransformerEncoder(
        input_dim=enc_cfg.get('input_dim', 49),
        model_dim=enc_cfg.get('model_dim', 128),
        num_heads=enc_cfg.get('num_heads', 4),
        num_layers=enc_cfg.get('num_layers', 4),
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    shots_list = config.get('fewshot', {}).get('shots', [5, 10, 20, 50])
    results = {}

    for ds_info in config['data']['datasets']:
        name = ds_info['name']
        ds_results = {}

        for shots in shots_list:
            adapt_path = f"data/splits/protocol_c/holdout_{name}/adapt_{shots}shot.parquet"
            test_path = f"data/splits/protocol_c/holdout_{name}/test_{shots}shot.parquet"

            if not os.path.exists(adapt_path) or not os.path.exists(test_path):
                print(f"Skipping {name}/{shots}-shot: split files not found")
                continue

            adapt_loader = create_dataloader(adapt_path, batch_size=min(shots * 2, 64), shuffle=True)
            test_loader = create_dataloader(test_path, batch_size=1024, shuffle=False)

            encoder_copy = copy.deepcopy(encoder)
            metrics = head_finetune(encoder_copy, adapt_loader, test_loader, config, device)

            ds_results[shots] = metrics
            print(f"{name} {shots}-shot: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

        results[name] = ds_results

    return results
