"""Self-supervised contrastive pre-training loop (Phase 2)."""

import os
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.config import load_config, get_device
from src.models.transformer_encoder import FlowTransformerEncoder
from src.models.projection_head import ProjectionHead
from src.models.classification_head import ClassificationHead
from src.data.augmentations import FlowAugmentor
from src.data.dataset import create_unlabeled_dataloader, create_dataloader
from src.training.losses import NTXentLoss
from src.training.schedulers import CosineWarmupScheduler


def train_contrastive(config_path: str = "configs/phase2_contrastive.yaml") -> dict:
    """Full contrastive pre-training loop."""
    config = load_config(config_path)
    device = get_device(config)

    # Build model
    enc_cfg = config['model']['encoder']
    encoder = FlowTransformerEncoder(
        input_dim=enc_cfg.get('input_dim', 49),
        model_dim=enc_cfg.get('model_dim', 128),
        num_heads=enc_cfg.get('num_heads', 4),
        num_layers=enc_cfg.get('num_layers', 4),
        feedforward_dim=enc_cfg.get('feedforward_dim', 512),
        dropout=enc_cfg.get('dropout', 0.1),
    ).to(device)

    proj_cfg = config['model']['projection_head']
    projection_head = ProjectionHead(
        input_dim=enc_cfg.get('model_dim', 128),
        hidden_dim=proj_cfg.get('hidden_dim', 64),
        output_dim=proj_cfg.get('output_dim', 32),
    ).to(device)

    # Augmentor
    aug_cfg = config['training']['contrastive']['augmentations']
    augmentor = FlowAugmentor(aug_cfg)

    # DataLoader
    data_path = os.path.join(config['data']['processed_dir'], 'combined_unlabeled.parquet')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing: {data_path}. Run preprocessing first.")

    train_cfg = config['training']
    dataloader = create_unlabeled_dataloader(data_path, batch_size=train_cfg.get('batch_size', 4096))

    # Loss, optimizer, scheduler
    contrastive_cfg = train_cfg['contrastive']
    criterion = NTXentLoss(temperature=contrastive_cfg.get('temperature', 0.07))

    opt_cfg = train_cfg['optimizer']
    params = list(encoder.parameters()) + list(projection_head.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=opt_cfg.get('lr', 0.0005), weight_decay=opt_cfg.get('weight_decay', 0.0001)
    )

    sched_cfg = train_cfg['scheduler']
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=sched_cfg.get('warmup_epochs', 10),
        total_epochs=sched_cfg.get('T_max', 200),
    )

    # Mixed precision
    use_amp = config['project'].get('precision', 'fp16') == 'fp16' and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # Checkpoint
    ckpt_dir = "checkpoints/phase2/"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume if available
    start_epoch = 0
    latest_path = os.path.join(ckpt_dir, 'latest.pt')
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        projection_head.load_state_dict(ckpt['projection_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    num_epochs = train_cfg.get('epochs', 200)
    monitor_cfg = train_cfg.get('monitoring', {})
    probe_every = monitor_cfg.get('linear_probe_every', 10)
    save_every = monitor_cfg.get('save_checkpoint_every', 20)

    best_probe_f1 = 0.0
    history = {'loss': [], 'probe_f1': []}

    for epoch in range(start_epoch, num_epochs):
        encoder.train()
        projection_head.train()
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            x = batch.to(device)

            # Create two augmented views
            view1 = augmentor(x)
            view2 = augmentor(x)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                z1 = projection_head(encoder(view1))
                z2 = projection_head(encoder(view2))
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            state = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projection_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(state, os.path.join(ckpt_dir, f'checkpoint_epoch{epoch+1}.pt'))
            torch.save(state, latest_path)

        # Linear probe evaluation
        if (epoch + 1) % probe_every == 0:
            probe_f1 = _run_linear_probe(encoder, config, device)
            history['probe_f1'].append(probe_f1)
            print(f"  Linear probe F1: {probe_f1:.4f}")

            if probe_f1 > best_probe_f1:
                best_probe_f1 = probe_f1
                torch.save(encoder.state_dict(), os.path.join(ckpt_dir, 'best_encoder.pt'))
                torch.save(projection_head.state_dict(), os.path.join(ckpt_dir, 'projection_head.pt'))
                print(f"  New best! Saved encoder.")

    # Save final
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, 'final_encoder.pt'))
    torch.save(projection_head.state_dict(), os.path.join(ckpt_dir, 'final_projection_head.pt'))

    return history


@torch.no_grad()
def _run_linear_probe(encoder, config, device):
    """
    Train a linear classifier on frozen encoder features using 1% of labeled data.
    Quick evaluation of representation quality.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import numpy as np

    encoder.eval()

    # Use first dataset's protocol_a split for probing
    splits_dir = os.path.join(config['data']['splits_dir'], 'protocol_a')
    first_ds = config['data']['datasets'][0]['name']

    train_path = os.path.join(splits_dir, f"{first_ds}_train.parquet")
    test_path = os.path.join(splits_dir, f"{first_ds}_test.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return 0.0

    # Sample 1% of training data
    frac = config['training'].get('monitoring', {}).get('linear_probe_label_fraction', 0.01)

    train_loader = create_dataloader(train_path, batch_size=2048, shuffle=False)
    test_loader = create_dataloader(test_path, batch_size=2048, shuffle=False)

    def extract_features(loader, max_samples=None):
        feats, labels = [], []
        count = 0
        for x, y in loader:
            x = x.to(device)
            f = encoder(x).cpu().numpy()
            feats.append(f)
            labels.append(y.numpy())
            count += x.shape[0]
            if max_samples and count >= max_samples:
                break
        return np.concatenate(feats), np.concatenate(labels)

    # Get total train size estimate
    total_train = len(train_loader.dataset)
    max_train = max(int(total_train * frac), 100)

    X_train, y_train = extract_features(train_loader, max_samples=max_train)
    X_test, y_test = extract_features(test_loader, max_samples=5000)

    clf = LogisticRegression(max_iter=200, solver='lbfgs', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return f1_score(y_test, y_pred, average='macro', zero_division=0)
