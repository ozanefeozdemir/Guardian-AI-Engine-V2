"""Federated learning integration using Flower framework."""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.config import load_config, get_device
from src.models.flowguard import FlowGuard
from src.data.federated_loader import create_federated_loaders


def get_parameters(model):
    """Extract model parameters as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Set model parameters from list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FlowGuardClient:
    """
    Flower client representing one network silo.
    Implements fit/evaluate for federated training.
    """

    def __init__(self, dataset_name: str, model: nn.Module,
                 train_loader, val_loader, config: dict, device=None):
        self.dataset_name = dataset_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or get_device(config)

    def get_parameters(self):
        return get_parameters(self.model)

    def fit(self, parameters):
        """Train locally and return updated parameters."""
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        self.model.train()

        client_cfg = self.config.get('federated', {}).get('client', {})
        local_epochs = client_cfg.get('local_epochs', 1)
        opt_cfg = client_cfg.get('optimizer', {})
        grad_clip = client_cfg.get('gradient_clip_norm', 1.0)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=opt_cfg.get('lr', 0.0001),
        )
        criterion = nn.CrossEntropyLoss()

        use_amp = self.device.type == 'cuda'
        scaler = GradScaler(enabled=use_amp)

        total_samples = 0
        for epoch in range(local_epochs):
            pbar = tqdm(self.train_loader, desc=f"  {self.dataset_name} epoch {epoch+1}/{local_epochs}", leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    logits = self.model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                total_samples += x.size(0)

        return get_parameters(self.model), total_samples

    @torch.no_grad()
    def evaluate(self, parameters):
        """Evaluate global model on local validation data."""
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        return total_loss / max(total, 1), correct / max(total, 1), total


def run_federated_simulation(config_path: str = "configs/phase3_federated.yaml") -> dict:
    """
    Run federated learning simulation.
    All 4 clients on same machine (no network communication).
    Implements FedProx manually since we may not have Flower installed.
    """
    config = load_config(config_path)
    device = get_device(config)
    fed_cfg = config.get('federated', {})

    # Build global model
    model = FlowGuard(config).to(device)

    # Load pre-trained encoder if available
    pretrained_path = config.get('model', {}).get('pretrained_encoder',
                      config.get('federated', {}).get('model', {}).get('pretrained_encoder', ''))
    if pretrained_path and os.path.exists(pretrained_path):
        encoder_state = torch.load(pretrained_path, map_location=device)
        model.encoder.load_state_dict(encoder_state)
        print(f"Loaded pre-trained encoder from {pretrained_path}")

    # Create per-silo loaders
    loaders = create_federated_loaders(config)

    # Create clients
    import copy
    clients = {}
    for name, silo_loaders in loaders.items():
        client_model = copy.deepcopy(model)
        clients[name] = FlowGuardClient(
            dataset_name=name,
            model=client_model,
            train_loader=silo_loaders.get('train'),
            val_loader=silo_loaders.get('val'),
            config=config,
            device=device,
        )

    # Federated training loop
    num_rounds = fed_cfg.get('server', {}).get('num_rounds', 100)
    mu = fed_cfg.get('server', {}).get('fedprox_mu', 0.01)

    ckpt_dir = "checkpoints/phase3/"
    os.makedirs(ckpt_dir, exist_ok=True)

    global_params = get_parameters(model)
    history = {'rounds': [], 'client_metrics': {name: [] for name in clients}}

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        # Local training
        client_updates = []
        client_sizes = []

        for name, client in clients.items():
            updated_params, n_samples = client.fit(global_params)
            client_updates.append(updated_params)
            client_sizes.append(n_samples)

        # FedAvg aggregation (weighted by number of samples)
        total_samples = sum(client_sizes)
        new_global_params = []
        for param_idx in range(len(global_params)):
            weighted_sum = sum(
                client_updates[i][param_idx] * (client_sizes[i] / total_samples)
                for i in range(len(clients))
            )
            new_global_params.append(weighted_sum)

        global_params = new_global_params

        # Evaluate
        for name, client in clients.items():
            loss, acc, n = client.evaluate(global_params)
            history['client_metrics'][name].append({'loss': loss, 'accuracy': acc})
            print(f"  {name}: loss={loss:.4f}, acc={acc:.4f}")

        history['rounds'].append(round_num + 1)

        # Save checkpoint every 10 rounds
        if (round_num + 1) % 10 == 0:
            set_parameters(model, global_params)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'global_round{round_num+1}.pt'))

    # Save final
    set_parameters(model, global_params)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final_global.pt'))

    return history
