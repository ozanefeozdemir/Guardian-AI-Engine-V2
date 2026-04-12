"""
End-to-end smoke test for FlowGuard pipeline.

Samples 5000 rows from each real raw CSV, then runs every pipeline stage
with minimal epochs/rounds to verify nothing crashes before cloud execution.

Usage:
    cd /path/to/guardian-engine-d3
    python scripts/smoke_test.py

All artefacts are written under data/smoke/ and checkpoints/smoke/.
"""

import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, get_device
from src.data.features import get_input_dim
from src.data.preprocess import preprocess_dataset, create_combined_unlabeled, DATASET_CATALOG
from src.data.splits import (
    generate_protocol_a_splits,
    generate_protocol_b_splits,
    generate_protocol_c_splits,
)
from src.data.dataset import create_dataloader, create_unlabeled_dataloader
from src.data.augmentations import FlowAugmentor
from src.models.mlp_baseline import MLPBaseline
from src.models.transformer_encoder import FlowTransformerEncoder
from src.models.projection_head import ProjectionHead
from src.models.flowguard import FlowGuard
from src.training.supervised import SupervisedTrainer
from src.training.losses import NTXentLoss
from src.training.contrastive import _run_linear_probe
from src.training.federated import FlowGuardClient, get_parameters, set_parameters
from src.training.fewshot import head_finetune
from src.training.adversarial import adversarial_training_step, domain_adversarial_training_step
from src.training.ewc import compute_and_save_fisher
from src.evaluation.metrics import evaluate_model

# ─── Constants ───────────────────────────────────────────────────────────────
SMOKE_DIR = "data/smoke"
RAW_SAMPLE_DIR = os.path.join(SMOKE_DIR, "raw")
PROCESSED_DIR = os.path.join(SMOKE_DIR, "processed")
SPLITS_DIR = os.path.join(SMOKE_DIR, "splits")
CKPT_DIR = "checkpoints/smoke"
N_ROWS = 5000           # rows sampled from each raw CSV
N_WORKERS = 0           # no multiprocessing in smoke test (avoids re-import)

# ─── Helpers ─────────────────────────────────────────────────────────────────
RESULTS: list[tuple[str, bool, str]] = []


def section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print("=" * 62)


def check(label: str, passed: bool, detail: str = "") -> bool:
    tag = "  [PASS]" if passed else "  [FAIL]"
    print(f"{tag}  {label}" + (f"  — {detail}" if detail else ""))
    RESULTS.append((label, passed, detail))
    return passed


def smoke_config(base: dict) -> dict:
    """Return a copy of *base* with all paths redirected to smoke dirs."""
    cfg = copy.deepcopy(base)
    cfg["data"]["raw_dir"] = RAW_SAMPLE_DIR
    cfg["data"]["processed_dir"] = PROCESSED_DIR
    cfg["data"]["splits_dir"] = SPLITS_DIR
    # run_full_preprocessing reads from config["paths"] with fallback to defaults;
    # patch both spellings so every helper finds the right dirs.
    cfg["paths"] = {
        "raw_data": RAW_SAMPLE_DIR,
        "processed_data": PROCESSED_DIR,
    }
    return cfg


def main() -> None:
    for d in [RAW_SAMPLE_DIR, PROCESSED_DIR, SPLITS_DIR, CKPT_DIR]:
        os.makedirs(d, exist_ok=True)

    # ─── Stage 0: sample raw CSVs ────────────────────────────────────────────────
section("Stage 0 · Sample raw CSVs")

RAW_DIR = "data/raw"
sampled: list[str] = []

for short_name, filename in DATASET_CATALOG.items():
    raw_path = os.path.join(RAW_DIR, filename)
    out_path = os.path.join(RAW_SAMPLE_DIR, filename)

    if os.path.exists(out_path):
        check(f"Sample {short_name} (cached)", True, out_path)
        sampled.append(short_name)
        continue

    if not os.path.exists(raw_path):
        check(f"Sample {short_name}", False, f"raw file not found: {raw_path}")
        continue

    try:
        t0 = time.time()
        df = pd.read_csv(raw_path, nrows=N_ROWS, low_memory=False)
        df.to_csv(out_path, index=False)
        check(f"Sample {short_name}", True, f"{len(df):,} rows in {time.time()-t0:.1f}s")
        sampled.append(short_name)
    except Exception as exc:
        check(f"Sample {short_name}", False, str(exc))

if not sampled:
    print("\n[ABORT] No raw CSVs could be sampled. Exiting.")
    sys.exit(1)

# ─── Stage 1: Preprocessing ──────────────────────────────────────────────────
section("Stage 1 · Preprocessing")

base_cfg = load_config("configs/base.yaml")
s_cfg = smoke_config(base_cfg)

processed: list[str] = []
first_stats = None

for short_name in sampled:
    filename = DATASET_CATALOG[short_name]
    raw_path = os.path.join(RAW_SAMPLE_DIR, filename)
    out_path = os.path.join(PROCESSED_DIR, f"{short_name}.parquet")

    if os.path.exists(out_path):
        check(f"Preprocess {short_name} (cached)", True)
        processed.append(short_name)
        if first_stats is None:
            from src.data.preprocess import PreprocessingStats
            stats_path = os.path.join(PROCESSED_DIR, f"{short_name}_stats.npz")
            if os.path.exists(stats_path):
                first_stats = PreprocessingStats.load(stats_path)
        continue

    try:
        stats = preprocess_dataset(
            raw_path=raw_path,
            output_path=out_path,
            fit_stats=True,
        )
        stats.save(os.path.join(PROCESSED_DIR, f"{short_name}_stats.npz"))
        df = pd.read_parquet(out_path)
        check(
            f"Preprocess {short_name}",
            len(df) > 0 and len(stats.feature_names) == get_input_dim(),
            f"{len(df):,} rows, {len(stats.feature_names)} features",
        )
        processed.append(short_name)
        if first_stats is None:
            first_stats = stats
    except Exception as exc:
        check(f"Preprocess {short_name}", False, str(exc))

if not processed:
    print("\n[ABORT] Preprocessing produced no output. Exiting.")
    sys.exit(1)

# Combined unlabeled
unlabeled_path = os.path.join(PROCESSED_DIR, "combined_unlabeled.parquet")
if not os.path.exists(unlabeled_path):
    try:
        create_combined_unlabeled(PROCESSED_DIR, processed)
        df_ul = pd.read_parquet(unlabeled_path)
        check("Combined unlabeled", len(df_ul) > 0, f"{len(df_ul):,} rows")
    except Exception as exc:
        check("Combined unlabeled", False, str(exc))
else:
    check("Combined unlabeled (cached)", True)

# ─── Stage 2: Splits ─────────────────────────────────────────────────────────
section("Stage 2 · Split generation")

splits_ok = False
try:
    generate_protocol_a_splits(s_cfg)
    # Verify at least one split file exists
    first = processed[0]
    train_path = os.path.join(SPLITS_DIR, "protocol_a", f"{first}_train.parquet")
    check("Protocol A splits", os.path.exists(train_path))

    if len(processed) >= 2:
        generate_protocol_b_splits(s_cfg)
        b_path = os.path.join(SPLITS_DIR, "protocol_b", f"holdout_{processed[0]}", "train.parquet")
        check("Protocol B splits", os.path.exists(b_path))

        generate_protocol_c_splits(s_cfg)
        c_path = os.path.join(SPLITS_DIR, "protocol_c", f"holdout_{processed[0]}", "adapt_5shot.parquet")
        check("Protocol C splits", os.path.exists(c_path))
    else:
        check("Protocol B/C splits", True, "skipped — need ≥2 datasets")

    splits_ok = True
except Exception as exc:
    check("Split generation", False, str(exc))

# ─── Stage 3: Phase 1 — MLP Baseline ─────────────────────────────────────────
section("Stage 3 · Phase 1: MLP Baseline (2 epochs)")

p1_ckpt_dir = os.path.join(CKPT_DIR, "phase1", processed[0])
os.makedirs(p1_ckpt_dir, exist_ok=True)

train_path = os.path.join(SPLITS_DIR, "protocol_a", f"{processed[0]}_train.parquet")
val_path = os.path.join(SPLITS_DIR, "protocol_a", f"{processed[0]}_val.parquet")

if splits_ok and os.path.exists(train_path) and os.path.exists(val_path):
    try:
        p1_cfg = load_config("configs/phase1_baseline.yaml")
        p1_cfg = smoke_config(p1_cfg)
        p1_cfg["training"]["epochs"] = 2
        p1_cfg["training"]["early_stopping"] = {"patience": 99, "metric": "f1_macro", "mode": "max"}

        train_loader = create_dataloader(train_path, batch_size=256, balanced=True, num_workers=N_WORKERS)
        val_loader = create_dataloader(val_path, batch_size=256, shuffle=False, num_workers=N_WORKERS)

        model_p1 = MLPBaseline(input_dim=get_input_dim(), hidden_dims=[128, 64], num_classes=2, dropout=0.3)
        trainer = SupervisedTrainer(model_p1, train_loader, val_loader, p1_cfg, checkpoint_dir=p1_ckpt_dir)
        history = trainer.train(num_epochs=2)

        ckpt_ok = os.path.exists(os.path.join(p1_ckpt_dir, "best_model.pt"))
        check("MLP Baseline train (2 epochs)", len(history["train_loss"]) == 2 and ckpt_ok,
              f"final val_f1={history['val_f1'][-1]:.4f}")
    except Exception as exc:
        check("MLP Baseline train (2 epochs)", False, str(exc))
else:
    check("MLP Baseline train", False, "splits not available")

# ─── Stage 4: Phase 2 — Contrastive Pretraining ──────────────────────────────
section("Stage 4 · Phase 2: Contrastive Pretraining (2 epochs)")

p2_ckpt_dir = os.path.join(CKPT_DIR, "phase2")
os.makedirs(p2_ckpt_dir, exist_ok=True)

device = get_device(base_cfg)

if os.path.exists(unlabeled_path):
    try:
        p2_cfg = load_config("configs/phase2_contrastive.yaml")
        p2_cfg = smoke_config(p2_cfg)
        p2_cfg["training"]["epochs"] = 2
        p2_cfg["training"]["monitoring"]["linear_probe_every"] = 2
        p2_cfg["training"]["monitoring"]["save_checkpoint_every"] = 2

        enc_cfg = p2_cfg["model"]["encoder"]
        encoder = FlowTransformerEncoder(
            input_dim=enc_cfg["input_dim"],
            model_dim=enc_cfg["model_dim"],
            num_heads=enc_cfg["num_heads"],
            num_layers=enc_cfg["num_layers"],
            feedforward_dim=enc_cfg["feedforward_dim"],
            dropout=enc_cfg["dropout"],
        ).to(device)

        proj_cfg = p2_cfg["model"]["projection_head"]
        projection_head = ProjectionHead(
            input_dim=enc_cfg["model_dim"],
            hidden_dim=proj_cfg["hidden_dim"],
            output_dim=proj_cfg["output_dim"],
        ).to(device)

        aug_cfg = p2_cfg["training"]["contrastive"]["augmentations"]
        augmentor = FlowAugmentor(aug_cfg)
        criterion = NTXentLoss(temperature=p2_cfg["training"]["contrastive"]["temperature"])

        from src.training.schedulers import CosineWarmupScheduler
        opt_cfg = p2_cfg["training"]["optimizer"]
        params = list(encoder.parameters()) + list(projection_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=2, total_epochs=2)

        dataloader = create_unlabeled_dataloader(unlabeled_path, batch_size=128, num_workers=N_WORKERS)

        losses = []
        for epoch in range(2):
            encoder.train(); projection_head.train()
            epoch_loss = 0; n = 0
            for batch in dataloader:
                x = batch.to(device)
                view1, view2 = augmentor(x), augmentor(x)
                optimizer.zero_grad()
                z1 = projection_head(encoder(view1))
                z2 = projection_head(encoder(view2))
                loss = criterion(z1, z2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item(); n += 1
            scheduler.step()
            losses.append(epoch_loss / max(n, 1))
            print(f"  Epoch {epoch+1}/2 | Loss: {losses[-1]:.4f}")

        torch.save(encoder.state_dict(), os.path.join(p2_ckpt_dir, "best_encoder.pt"))
        torch.save(projection_head.state_dict(), os.path.join(p2_ckpt_dir, "projection_head.pt"))
        check("Contrastive pretraining (2 epochs)", not any(np.isnan(losses)),
              f"final loss={losses[-1]:.4f}")
    except Exception as exc:
        check("Contrastive pretraining (2 epochs)", False, str(exc))
        encoder = None
else:
    check("Contrastive pretraining", False, "combined_unlabeled.parquet not found")
    encoder = None

# ─── Stage 5: Phase 3 — Federated Learning ───────────────────────────────────
section("Stage 5 · Phase 3: Federated Learning (2 rounds)")

p3_ckpt_dir = os.path.join(CKPT_DIR, "phase3")
os.makedirs(p3_ckpt_dir, exist_ok=True)

if splits_ok:
    try:
        p3_cfg = load_config("configs/phase3_federated.yaml")
        p3_cfg = smoke_config(p3_cfg)
        p3_cfg["federated"]["server"]["num_rounds"] = 2
        p3_cfg["federated"]["client"]["local_epochs"] = 1
        p3_cfg["federated"]["client"]["batch_size"] = 128
        p3_cfg["federated"]["client"]["num_workers"] = 0

        global_model = FlowGuard(p3_cfg).to(device)

        from src.data.federated_loader import create_federated_loaders
        loaders = create_federated_loaders(p3_cfg)

        if not loaders:
            check("Federated learning (2 rounds)", False, "no silo loaders created")
        else:
            import copy as _copy
            clients = {}
            for name, silo in loaders.items():
                if "train" not in silo or "val" not in silo:
                    continue
                client_model = _copy.deepcopy(global_model)
                clients[name] = FlowGuardClient(
                    dataset_name=name,
                    model=client_model,
                    train_loader=silo["train"],
                    val_loader=silo["val"],
                    config=p3_cfg,
                    device=device,
                )

            global_params = get_parameters(global_model)
            round_accs = []

            for round_num in range(2):
                updates, sizes = [], []
                for client in clients.values():
                    upd, n = client.fit(global_params)
                    updates.append(upd); sizes.append(n)

                total = sum(sizes)
                new_params = []
                for pi in range(len(global_params)):
                    new_params.append(sum(updates[i][pi] * (sizes[i] / total) for i in range(len(updates))))
                global_params = new_params

                accs = []
                for client in clients.values():
                    _, acc, _ = client.evaluate(global_params)
                    accs.append(acc)
                round_accs.append(np.mean(accs))
                print(f"  Round {round_num+1}/2 | mean acc={round_accs[-1]:.4f}")

            set_parameters(global_model, global_params)
            torch.save(global_model.state_dict(), os.path.join(p3_ckpt_dir, "final_global.pt"))
            check("Federated learning (2 rounds)", True,
                  f"clients={len(clients)}, final acc={round_accs[-1]:.4f}")
    except Exception as exc:
        check("Federated learning (2 rounds)", False, str(exc))
else:
    check("Federated learning", False, "splits not available")

# ─── Stage 6: Phase 4 — Few-Shot Adaptation ──────────────────────────────────
section("Stage 6 · Phase 4: Few-Shot Adaptation")

if encoder is not None and splits_ok and len(processed) >= 2:
    try:
        p4_cfg = load_config("configs/phase4_fewshot.yaml")
        p4_cfg = smoke_config(p4_cfg)

        holdout = processed[0]
        adapt_path = os.path.join(SPLITS_DIR, "protocol_c", f"holdout_{holdout}", "adapt_5shot.parquet")
        test_path = os.path.join(SPLITS_DIR, "protocol_c", f"holdout_{holdout}", "test_5shot.parquet")

        if os.path.exists(adapt_path) and os.path.exists(test_path):
            import copy as _copy
            enc_copy = _copy.deepcopy(encoder)
            adapt_loader = create_dataloader(adapt_path, batch_size=16, shuffle=True, num_workers=N_WORKERS)
            test_loader = create_dataloader(test_path, batch_size=256, shuffle=False, num_workers=N_WORKERS)

            fs_cfg_inner = {"head_finetune": {"lr": 0.001, "epochs": 5, "early_stopping_patience": 99}}
            p4_cfg["fewshot"] = fs_cfg_inner
            metrics = head_finetune(enc_copy, adapt_loader, test_loader, p4_cfg, device)
            check("Few-shot adaptation (5-shot)", True,
                  f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")
        else:
            check("Few-shot adaptation", False, "protocol_c splits not found")
    except Exception as exc:
        check("Few-shot adaptation", False, str(exc))
elif encoder is None:
    check("Few-shot adaptation", False, "encoder not trained (phase 2 failed)")
else:
    check("Few-shot adaptation", False, "need ≥2 datasets for protocol C")

# ─── Stage 7: Phase 5 — Hardening ────────────────────────────────────────────
section("Stage 7 · Phase 5: Hardening (domain adversarial + EWC)")

if splits_ok and os.path.exists(train_path):
    # --- Domain adversarial ---
    try:
        p5_cfg = load_config("configs/phase5_hardening.yaml")
        p5_cfg = smoke_config(p5_cfg)

        hard_model = FlowGuard(p5_cfg)
        hard_model.enable_domain_discriminator(num_domains=len(processed))
        hard_model.to(device)  # move after adding submodule so all parts land on device

        optimizer_h = torch.optim.Adam(hard_model.parameters(), lr=1e-4)
        criterion_h = torch.nn.CrossEntropyLoss()

        total_loss = 0; n_steps = 0
        for ds_idx, ds_name in enumerate(processed):
            ds_train_path = os.path.join(SPLITS_DIR, "protocol_a", f"{ds_name}_train.parquet")
            if not os.path.exists(ds_train_path):
                continue
            loader = create_dataloader(ds_train_path, batch_size=64, num_workers=N_WORKERS)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                domain_labels = torch.full((x.size(0),), ds_idx, dtype=torch.long, device=device)
                loss, _ = domain_adversarial_training_step(
                    hard_model, x, y, domain_labels, optimizer_h, criterion_h, lambda_grl=0.5)
                total_loss += loss; n_steps += 1
                if n_steps >= 5:
                    break
            if n_steps >= 5:
                break

        check("Domain adversarial step", n_steps > 0 and not np.isnan(total_loss / max(n_steps, 1)),
              f"{n_steps} steps, mean loss={total_loss/max(n_steps,1):.4f}")
    except Exception as exc:
        check("Domain adversarial step", False, str(exc))

    # --- EWC Fisher ---
    try:
        ewc_model = FlowGuard(p5_cfg).to(device)
        loader_ewc = create_dataloader(train_path, batch_size=64, num_workers=N_WORKERS)
        fisher_path = os.path.join(CKPT_DIR, "phase5", "fisher.pt")
        os.makedirs(os.path.join(CKPT_DIR, "phase5"), exist_ok=True)
        compute_and_save_fisher(ewc_model, loader_ewc, fisher_path, num_samples=128, device=device)
        check("EWC Fisher computation", os.path.exists(fisher_path))
    except Exception as exc:
        check("EWC Fisher computation", False, str(exc))

    # --- PGD adversarial training step ---
    try:
        pgd_model = FlowGuard(p5_cfg).to(device)
        optimizer_pgd = torch.optim.Adam(pgd_model.parameters(), lr=1e-4)
        criterion_pgd = torch.nn.CrossEntropyLoss()
        loader_pgd = create_dataloader(train_path, batch_size=32, num_workers=N_WORKERS)
        x_batch, y_batch = next(iter(loader_pgd))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        loss_val = adversarial_training_step(
            pgd_model, x_batch, y_batch, optimizer_pgd, criterion_pgd,
            epsilon=0.05, pgd_steps=2, step_size=0.025)
        check("PGD adversarial step", not np.isnan(loss_val), f"loss={loss_val:.4f}")
    except Exception as exc:
        check("PGD adversarial step", False, str(exc))
else:
    check("Hardening", False, "splits not available")

# ─── Stage 8: Evaluation ─────────────────────────────────────────────────────
section("Stage 8 · Evaluation")

test_p = os.path.join(SPLITS_DIR, "protocol_a", f"{processed[0]}_test.parquet")
if splits_ok and os.path.exists(test_p):
    # Evaluate MLP baseline
    try:
        ckpt_path = os.path.join(p1_ckpt_dir, "best_model.pt")
        eval_model = MLPBaseline(input_dim=get_input_dim(), hidden_dims=[128, 64], num_classes=2)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            eval_model.load_state_dict(ckpt["model_state_dict"])
        eval_model.to(device)
        test_loader = create_dataloader(test_p, batch_size=256, shuffle=False, num_workers=N_WORKERS)
        metrics = evaluate_model(eval_model, test_loader, device)
        check("Protocol A evaluation (MLP)", True,
              f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}, fpr={metrics.get('fpr', 0):.4f}")
    except Exception as exc:
        check("Protocol A evaluation (MLP)", False, str(exc))

    # Evaluate FlowGuard (federated)
    if os.path.exists(os.path.join(p3_ckpt_dir, "final_global.pt")):
        try:
            fg_model = FlowGuard(p3_cfg).to(device)
            fg_model.load_state_dict(torch.load(os.path.join(p3_ckpt_dir, "final_global.pt"), map_location=device))
            metrics_fg = evaluate_model(fg_model, test_loader, device)
            check("Protocol A evaluation (FlowGuard)", True,
                  f"acc={metrics_fg['accuracy']:.4f}, f1={metrics_fg['f1_macro']:.4f}")
        except Exception as exc:
            check("Protocol A evaluation (FlowGuard)", False, str(exc))
    else:
        check("Protocol A evaluation (FlowGuard)", True, "skipped — no checkpoint (expected if phase 3 failed)")
else:
    check("Evaluation", False, "splits not available")

# ─── Summary ─────────────────────────────────────────────────────────────────
section("Summary")

total = len(RESULTS)
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = [(label, detail) for label, ok, detail in RESULTS if not ok]

print(f"\n  {passed}/{total} checks passed")

if failed:
    print("\n  Failed checks:")
    for label, detail in failed:
        print(f"    ✗  {label}" + (f": {detail}" if detail else ""))
else:
    print("\n  All checks passed — pipeline is ready for cloud execution.")

print()
