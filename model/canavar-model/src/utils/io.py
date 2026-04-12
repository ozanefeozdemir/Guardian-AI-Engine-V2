"""Save/load checkpoints and results."""

import os
import json
import torch


class CheckpointManager:
    """
    Saves checkpoints to persistent storage (e.g., Google Drive).
    On resume, loads the latest checkpoint and continues training.

    Usage:
        ckpt = CheckpointManager("checkpoints/phase2/", resume=True)
        start_epoch = ckpt.load(model, optimizer, scheduler)
        for epoch in range(start_epoch, total_epochs):
            # ... train ...
            ckpt.save(model, optimizer, scheduler, epoch, metrics)
    """

    def __init__(self, checkpoint_dir: str, resume: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch: int,
             metrics: dict, is_best: bool = False) -> None:
        """Save checkpoint. If is_best, also save as best_model.pt."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
        }

        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(state, best_path)

    def load(self, model, optimizer=None, scheduler=None) -> int:
        """Load latest checkpoint. Returns the epoch to resume from."""
        if not self.resume:
            return 0

        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        if not os.path.exists(latest_path):
            return 0

        checkpoint = torch.load(latest_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {epoch}")
        return epoch

    def load_best(self, model) -> dict:
        """Load best checkpoint. Returns metrics."""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best checkpoint at {best_path}")

        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('metrics', {})


def save_results(results: dict, path: str):
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: str) -> dict:
    """Load results dict from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
