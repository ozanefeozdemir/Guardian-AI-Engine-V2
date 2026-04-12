"""Elastic Weight Consolidation for continual learning (Phase 5)."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.training.losses import EWCLoss


def compute_and_save_fisher(model, dataloader, save_path, num_samples=2000, device=None):
    """Compute Fisher Information Matrix and save with model parameters."""
    if device is None:
        device = next(model.parameters()).device

    fisher = EWCLoss.compute_fisher(model, dataloader, num_samples, device)
    old_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    torch.save({
        'fisher': fisher,
        'params': old_params,
    }, save_path)
    print(f"Saved Fisher matrix and parameters to {save_path}")
    return fisher, old_params


def load_ewc_loss(model, fisher_path, lambda_=5000):
    """Load saved Fisher matrix and create EWC loss."""
    data = torch.load(fisher_path, map_location=next(model.parameters()).device)
    return EWCLoss(model, data['params'], data['fisher'], lambda_)


def train_with_ewc(model, train_loader, val_loader, ewc_loss,
                   config, device, num_epochs=50):
    """Train model with EWC regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n = 0

        for x, y in tqdm(train_loader, desc=f"EWC Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            task_loss = criterion(logits, y)
            reg_loss = ewc_loss(model)
            loss = task_loss + reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/n:.4f}")

    return model
