# FlowGuard — Project Report

**Date:** 2026-04-09
**Model:** FlowGuard — Flow-based Network Intrusion Detection System
**Architecture:** Contrastive Pre-training + Federated Learning + Few-Shot Adaptation

---

## Datasets

| Dataset | Full Name | Size (raw) | Traffic Type |
|---------|-----------|------------|--------------|
| unsw | NF-UNSW-NB15-v3 | ~577 MB | Mixed benign/attack |
| bot | NF-BoT-IoT-v3 | ~2 GB | 99%+ attack traffic |
| ton | NF-ToN-IoT-v3 | ~5.3 GB | Mixed, many attack types |
| cic | NF-CICIDS2018-v3 | ~5 GB | Mixed, diverse attacks |

All datasets are NF-v3 format (NetFlow-based), sourced from the UQ NIDS dataset collection.

---

## Preprocessing

**Input features:** 53 (after feature engineering)
- Dropped 4 identity columns + 2 timestamp columns from raw data
- Added 6 port bucket features (well-known ports, ephemeral, etc.)
- Label encoding: binary (0=Benign, 1=Attack)

**Processed outputs** (`data/processed/`, 1.4 GB total):
- `unsw.parquet`, `bot.parquet`, `ton.parquet`, `cic.parquet`
- `combined_unlabeled.parquet` — 1,551,989 benign rows merged from all 4 datasets (capped at 500k/dataset), used for Phase 2 contrastive pretraining

**Split generation** (`data/splits/`):
- **Protocol A** (315 MB) — Within-dataset: train/val/test per dataset
- **Protocol B** (6.3 GB) — Cross-dataset leave-one-out: for each dataset, train on the other 3, test on the held-out one
- **Protocol C** (10 GB) — Few-shot: holdout splits with 5/10/20/50-shot adaptation sets per dataset

---

## Phase 1 — MLP Baseline

**Model:** MLPBaseline (feedforward network)
**Training:** Protocol A splits, per-dataset independently
**Hardware:** Google Colab A100

**Results (Protocol A — Within-dataset):**

| Dataset | Accuracy | F1 Macro | Notes |
|---------|----------|----------|-------|
| unsw | 100% | 100% | Test set all-benign (sequential split artifact) |
| bot | 2.87% | 2.85% | Trained before ClassBalancedSampler fix |
| ton | 82.5% | 67.4% | |
| cic | 99.4% | 99.4% | |

**Issues fixed during Phase 1:**
- `WeightedRandomSampler` hit PyTorch's 2²⁴ category limit on ToN-IoT (22M rows) → replaced with custom `ClassBalancedSampler`
- DataLoader worker crashes (`num_workers=2` exhausted `/dev/shm`) → set `num_workers=0`
- Bot trained before the sampler fix → underperformed

---

## Phase 2 — Contrastive Self-Supervised Pretraining

**Model:** FlowTransformerEncoder + ProjectionHead
**Architecture:**
- `FlowTransformerEncoder`: treats each of 53 features as a token, prepends CLS token → 54 tokens, positional embedding size 54×128
- `ProjectionHead`: maps 128-dim encoder output to 32-dim contrastive space
- Loss: NTXentLoss (SimCLR) with temperature=0.07

**Training:**
- Data: `combined_unlabeled.parquet` (1.55M benign rows, no labels)
- 200 epochs, batch size 4096, ~1515 steps/epoch
- Mixed precision (fp16/AMP) on A100
- CosineWarmup scheduler: warmup 10 epochs, peak LR 0.0005

**Loss progression:** 3.31 → 1.39 (epoch 55), continued decreasing
**Linear probe F1:** ~0.50 throughout (expected — pretrained on benign-only data, no attack signal)

**Checkpoints saved:**
- `checkpoints/phase2/best_encoder.pt` (best linear probe)
- `checkpoints/phase2/final_encoder.pt` (epoch 200)
- Intermediate checkpoints every 20 epochs

**Issues fixed:**
- `NTXentLoss.masked_fill_` inplace op with `-1e9` overflows fp16 (max ~65504) → changed to `sim.masked_fill(mask, torch.finfo(sim.dtype).min / 2)`
- Same fix applied to `SupervisedContrastiveLoss`
- `combined_unlabeled.parquet` missing (preprocessing step skipped) → created manually with 500k cap per dataset

---

## Phase 3 — Federated Learning (FedAvg)

**Model:** FlowGuard (FlowTransformerEncoder + ClassificationHead)
**Warm-start:** Phase 2 `best_encoder.pt`
**Protocol:** FedAvg simulation with 4 clients (one per dataset)

**Config:**
- Rounds: 10 (reduced from 50 for Colab feasibility)
- Local epochs: 2 (reduced from 5)
- Batch size: 512
- Balanced sampling: yes (ClassBalancedSampler)
- LR: 0.0001, AdamW

**Training samples per round:**

| Client | Samples/round |
|--------|--------------|
| unsw | 373,008 |
| bot | 399,924 |
| ton | 259,892 |
| cic | 343,116 |

**Final round (10/10) accuracy:**

| Client | Loss | Accuracy |
|--------|------|----------|
| unsw | 0.0003 | 99.99% |
| bot | 0.0061 | 99.88% |
| ton | 0.4533 | 89.02% |
| cic | 0.2075 | 94.29% |

**Protocol A evaluation (within-dataset):**

| Dataset | Accuracy | F1 Macro | AUROC | FPR |
|---------|----------|----------|-------|-----|
| unsw | 100% | 50%* | nan* | 0% |
| bot | 87.7% | 55.9% | 0.675 | 50.3% |
| ton | 94.8% | 92.2% | 0.983 | 4.4% |
| cic | 98.6% | 96.5% | 0.991 | 1.2% |

*unsw test set is all-benign — sequential split artifact, not a model issue.

**Protocol B evaluation (cross-dataset generalization):**

| Holdout | Accuracy | F1 Macro | AUROC | FPR |
|---------|----------|----------|-------|-----|
| unsw | 99.97% | 99.87% | 1.000 | 0.02% |
| bot | 98.75% | 59.79% | 0.968 | 48.3% |
| ton | 92.60% | 92.19% | 0.973 | 5.4% |
| cic | 97.96% | 95.57% | 0.982 | 1.5% |

**Bot weakness:** FPR ~50% in both protocols. Root cause: BoT-IoT training data is 99%+ attacks — the model never learned what normal bot traffic looks like. Phase 4 few-shot adaptation is designed to fix this.

**vs Phase 1 MLP Baseline (Protocol B):**
- ton: MLP F1 0.45 → FlowGuard F1 0.92 (+105%)
- cic: MLP F1 0.45 → FlowGuard F1 0.96 (+113%)

**Issues fixed:**
- `FlowGuardClient` had no progress visibility → added tqdm bars per client per epoch
- `.gitkeep` file in splits dir caused `ArrowInvalid: Parquet file size is 0 bytes` → deleted
- Training splits too large for Colab RAM (ton_train ~266MB) → subsampled to manageable size
- Round time ~2-3 hours → reduced `num_rounds` 50→10, `local_epochs` 5→2

---

## Phase 4 — Few-Shot Adaptation

**Status:** Config fixed, not yet fully evaluated
**Fix applied:** `phase4_fewshot.yaml` had wrong encoder config (`type: mlp`, `input_dim` missing) → corrected to match FlowTransformerEncoder with `input_dim: 53`, also fixed checkpoint path (`federated_best.pt` → `final_global.pt`)

---

## Phase 5 — Hardening

**Model:** FlowGuard with DomainDiscriminator
**Techniques:** EWC (Elastic Weight Consolidation) + adversarial training
**Checkpoint:** `checkpoints/phase5/hardened_model.pt` + `fisher.pt`

**Note:** Evaluation requires loading with `model.enable_domain_discriminator(num_domains=4)` before `load_state_dict` due to extra domain discriminator keys in the state dict.

---

## Key Bugs Fixed Across All Phases

| Bug | Location | Fix |
|-----|----------|-----|
| `masked_fill_` inplace fp16 overflow | `losses.py` NTXentLoss + SupConLoss | `masked_fill(mask, torch.finfo(sim.dtype).min / 2)` |
| Config inheritance broken | All phase configs | `defaults: - base` → `inherits: base.yaml` |
| Phase 2 config structural mismatch | `phase2_contrastive.yaml` | Full rewrite matching `train_contrastive()` expected keys |
| Phase 3 config key mismatch | `phase3_federated.yaml` | `client_config` → `client`, `rounds` → `server.num_rounds` |
| Phase 5 encoder `input_dim` missing | `phase5_hardening.yaml` | Added `input_dim: 53` |
| Phase 4 encoder config wrong | `phase4_fewshot.yaml` | Replaced MLP encoder config with TransformerEncoder config |
| `enable_domain_discriminator` device mismatch | `flowguard.py` | Auto-detect device from encoder parameters |
| `WeightedRandomSampler` 2²⁴ limit | `dataset.py` | Replaced with custom `ClassBalancedSampler` |
| DataLoader worker crash | `dataset.py`, `federated_loader.py` | `num_workers=0` |
| `ClassBalancedSampler` 40M Python objects | `dataset.py` | `.numpy()` + `max_per_class=500_000` cap |
| `combined_unlabeled.parquet` missing | Preprocessing | Created manually with 500k/dataset cap |
| Session crash resets working directory | Colab | `os.chdir("/content/flowguard")` at session start |

---

## Infrastructure Notes

- **Hardware:** Google Colab A100 (40GB VRAM, 84GB RAM)
- **Framework:** PyTorch with fp16 AMP (GradScaler + autocast)
- **Config system:** YAML with `inherits:` key for inheritance + `deep_merge`
- **Sampler:** Custom `ClassBalancedSampler` (avoids PyTorch multinomial limit, uses numpy for performance)
- **Checkpoints:** Saved per phase in `checkpoints/phase{N}/`
- **Splits:** Protocol A (315MB), B (6.3GB), C (10GB)
