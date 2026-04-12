"""Experiment logging utilities (W&B or local)."""

import os
import json
from datetime import datetime


class ExperimentLogger:
    """
    Lightweight experiment logger. Supports W&B (optional) and local JSON logging.
    """

    def __init__(self, project_name: str = "flowguard", experiment_name: str = None,
                 use_wandb: bool = False, log_dir: str = "results/logs"):
        self.project_name = project_name
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.history = []
        self.run = None

        if use_wandb:
            try:
                import wandb
                self.run = wandb.init(
                    project=project_name,
                    name=self.experiment_name,
                    reinit=True,
                )
            except Exception as e:
                print(f"W&B init failed: {e}. Falling back to local logging.")
                self.use_wandb = False

    def log(self, metrics: dict, step: int = None):
        """Log metrics for a given step."""
        entry = {"step": step, **metrics}
        self.history.append(entry)

        if self.use_wandb and self.run:
            import wandb
            wandb.log(metrics, step=step)

    def log_config(self, config: dict):
        """Log experiment configuration."""
        if self.use_wandb and self.run:
            import wandb
            wandb.config.update(config)

        config_path = os.path.join(self.log_dir, f"{self.experiment_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def save(self):
        """Save log history to JSON file."""
        log_path = os.path.join(self.log_dir, f"{self.experiment_name}_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def finish(self):
        """Finish logging session."""
        self.save()
        if self.use_wandb and self.run:
            import wandb
            wandb.finish()
