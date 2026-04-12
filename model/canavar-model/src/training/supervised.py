"""Standard supervised training loop."""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.config import get_device


class SupervisedTrainer:
    """
    Standard supervised training loop with mixed precision, early stopping,
    and checkpoint management.
    """

    def __init__(self, model, train_loader, val_loader, config: dict,
                 checkpoint_dir: str = "checkpoints/phase1/"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = get_device(config)
        self.model.to(self.device)

        # Training config
        train_cfg = config.get('training', {})
        opt_cfg = train_cfg.get('optimizer', {})

        # Optimizer
        opt_type = opt_cfg.get('type', 'adam').lower()
        lr = opt_cfg.get('lr', 0.001)
        wd = opt_cfg.get('weight_decay', 0.0001)

        if opt_type == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        sched_cfg = train_cfg.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'cosine')
        if sched_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched_cfg.get('T_max', train_cfg.get('epochs', 50))
            )
        elif sched_type == 'cosine_warmup':
            from src.training.schedulers import CosineWarmupScheduler
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=sched_cfg.get('warmup_epochs', 10),
                total_epochs=sched_cfg.get('T_max', train_cfg.get('epochs', 200)),
            )
        else:
            self.scheduler = None

        # Loss with optional class weights
        class_weights = train_cfg.get('class_weights', None)
        if class_weights == 'balanced':
            # Compute from training data
            all_labels = []
            for _, labels in train_loader:
                all_labels.append(labels)
            all_labels = torch.cat(all_labels)
            class_counts = torch.bincount(all_labels).float()
            weights = 1.0 / (class_counts + 1e-8)
            weights = weights / weights.sum() * len(weights)
            self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        use_fp16 = config.get('project', {}).get('precision', 'fp16') == 'fp16'
        self.scaler = GradScaler(enabled=use_fp16 and self.device.type == 'cuda')
        self.use_amp = use_fp16 and self.device.type == 'cuda'

        # Early stopping
        es_cfg = train_cfg.get('early_stopping', {})
        self.patience = es_cfg.get('patience', 5)
        self.es_metric = es_cfg.get('metric', 'f1_macro')
        self.es_mode = es_cfg.get('mode', 'max')

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_metric = -float('inf') if self.es_mode == 'max' else float('inf')
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(self.train_loader, desc="Training", leave=False):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro', zero_division=0)
        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())

        return total_loss / len(all_labels), acc, f1

    def train(self, num_epochs: int = None):
        if num_epochs is None:
            num_epochs = self.config.get('training', {}).get('epochs', 50)

        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()

            if self.scheduler:
                self.scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f}")

            # Early stopping check
            metric = val_f1 if self.es_metric == 'f1_macro' else val_acc
            improved = (metric > self.best_metric) if self.es_mode == 'max' else (metric < self.best_metric)

            if improved:
                self.best_metric = metric
                self.patience_counter = 0
                self._save_checkpoint(epoch, metric, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Regular checkpoint
            self._save_checkpoint(epoch, metric)

        return history

    def _save_checkpoint(self, epoch, metric, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(state, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(state, best_path)

    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('epoch', 0)
