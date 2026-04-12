"""Loss functions for all training phases."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (SimCLR).
    Given a batch of N samples with two views each, creates 2N samples.
    Each sample's positive pair is the other view of the same original.
    All other 2(N-1) samples are negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (N, D) embeddings of view 1 (already L2-normalized)
            z_j: (N, D) embeddings of view 2 (already L2-normalized)
        Returns:
            Scalar loss
        """
        N = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min / 2)

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_i = torch.arange(N, device=z.device)
        pos_j = pos_i + N

        # Labels: for sample i, positive is at index i+N; for sample i+N, positive is at index i
        labels = torch.cat([pos_j, pos_i], dim=0)  # (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    SupCon loss. Pulls same-class samples together, pushes different-class apart.
    Used in Phase 4 few-shot adaptation.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, D) L2-normalized embeddings
            labels: (N,) integer class labels
        """
        device = features.device
        N = features.shape[0]

        # Similarity matrix
        sim = torch.mm(features, features.t()) / self.temperature  # (N, N)

        # Mask: same class = 1, different class = 0, diagonal = 0
        labels = labels.unsqueeze(1)
        mask = (labels == labels.t()).float().to(device)
        mask.fill_diagonal_(0)

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # Mask out self
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min / 2)

        # Log-sum-exp of all non-self entries
        exp_sim = torch.exp(sim) * (~self_mask).float()
        log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)

        # Mean of log-prob over positive pairs
        pos_count = mask.sum(dim=1)
        # Avoid division by zero for samples with no positives
        has_pos = pos_count > 0

        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_sim = (sim * mask).sum(dim=1)  # Sum of similarities with positives
        mean_pos_sim = pos_sim[has_pos] / pos_count[has_pos]

        loss = (-mean_pos_sim + log_sum_exp[has_pos]).mean()
        return loss


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation loss.
    Penalizes changes to parameters important for previous tasks.

    Usage:
        ewc = EWCLoss(model, old_params, fisher_matrix, lambda_=5000)
        total_loss = task_loss + ewc(model)
    """

    def __init__(self, model: nn.Module, old_params: dict,
                 fisher_matrix: dict, lambda_: float = 5000):
        super().__init__()
        self.old_params = {k: v.clone().detach() for k, v in old_params.items()}
        self.fisher = {k: v.clone().detach() for k, v in fisher_matrix.items()}
        self.lambda_ = lambda_

    def forward(self, model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.old_params[name]) ** 2).sum()
        return 0.5 * self.lambda_ * loss

    @staticmethod
    def compute_fisher(model: nn.Module, dataloader, num_samples: int = 2000,
                       device: torch.device = None) -> dict:
        """Estimate diagonal Fisher Information Matrix."""
        if device is None:
            device = next(model.parameters()).device

        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()

        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break

            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                continue

            model.zero_grad()
            output = model(x)
            # Use log-likelihood of predicted class
            log_probs = F.log_softmax(output, dim=1)
            labels = output.argmax(dim=1)
            loss = F.nll_loss(log_probs, labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 * x.shape[0]

            count += x.shape[0]

        # Normalize
        for n in fisher:
            fisher[n] /= max(count, 1)

        return fisher
