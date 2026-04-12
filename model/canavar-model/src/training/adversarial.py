"""Domain adversarial training and PGD adversarial training (Phase 5)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


def pgd_attack(model, x, y, epsilon=0.1, step_size=0.025, num_steps=7, device=None):
    """
    Projected Gradient Descent adversarial attack.

    Args:
        model: Classification model
        x: Input tensor
        y: True labels
        epsilon: Maximum perturbation magnitude
        step_size: PGD step size
        num_steps: Number of PGD iterations
    Returns:
        Adversarial examples
    """
    if device is None:
        device = x.device

    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            perturbation = step_size * x_adv.grad.sign()
            x_adv = x_adv + perturbation
            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = (x + delta).detach()

    return x_adv


def adversarial_training_step(model, x, y, optimizer, criterion,
                               epsilon=0.1, pgd_steps=7, step_size=0.025,
                               scaler=None, use_amp=False):
    """
    Single adversarial training step: train on both clean and adversarial examples.
    """
    device = x.device

    # Generate adversarial examples
    model.eval()
    x_adv = pgd_attack(model, x, y, epsilon, step_size, pgd_steps, device)
    model.train()

    optimizer.zero_grad()

    if scaler and use_amp:
        with autocast(enabled=True):
            logits_clean = model(x)
            logits_adv = model(x_adv)
            loss = 0.5 * (criterion(logits_clean, y) + criterion(logits_adv, y))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits_clean = model(x)
        logits_adv = model(x_adv)
        loss = 0.5 * (criterion(logits_clean, y) + criterion(logits_adv, y))
        loss.backward()
        optimizer.step()

    return loss.item()


def domain_adversarial_training_step(model, x, y, domain_labels,
                                      optimizer, criterion, lambda_grl=1.0,
                                      scaler=None, use_amp=False):
    """
    Single domain adversarial training step.
    Model must have forward_with_domain method.
    """
    optimizer.zero_grad()

    if scaler and use_amp:
        with autocast(enabled=True):
            logits, domain_logits, _ = model.forward_with_domain(x, lambda_=lambda_grl)
            task_loss = criterion(logits, y)
            domain_loss = F.cross_entropy(domain_logits, domain_labels) if domain_logits is not None else 0
            loss = task_loss + domain_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, domain_logits, _ = model.forward_with_domain(x, lambda_=lambda_grl)
        task_loss = criterion(logits, y)
        domain_loss = F.cross_entropy(domain_logits, domain_labels) if domain_logits is not None else 0
        loss = task_loss + domain_loss
        loss.backward()
        optimizer.step()

    return loss.item(), task_loss.item()
