# FlowGuard: Generalizable Flow-Based Network Intrusion Detection System

A fully flow-based NIDS using deep learning that generalizes across unseen networks with minimal performance degradation.

## Approach

1. **Self-supervised contrastive pre-training** on standardized NetFlow features
2. **Federated learning** across heterogeneous network silos
3. **Few-shot adaptation** for new network deployment

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

## Data

**You must manually download and place the following files in `data/raw/`:**

| File | Source |
|------|--------|
| `NF-UNSW-NB15-v3.csv` | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |
| `NF-BoT-IoT-v3.csv` | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |
| `NF-ToN-IoT-v3.csv` | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |
| `NF-CSE-CIC-IDS2018-v3.csv` | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |

The code will **never** attempt to download data files.

## Pipeline

```bash
# 1. Validate raw data
make validate

# 2. Preprocess (clean, normalize, split)
make preprocess

# 3. Train MLP baseline (Phase 1)
make train-baseline

# 4. Contrastive pre-training (Phase 2)
make train-contrastive

# 5. Federated learning (Phase 3)
make train-federated

# 6. Few-shot adaptation (Phase 4)
make adapt

# 7. Evaluate
make evaluate
```

## Evaluation Protocols

- **Protocol A**: Within-dataset (sanity check)
- **Protocol B**: Cross-dataset, leave-one-out (generalization test)
- **Protocol C**: Few-shot adaptation (deployment simulation)

## Testing

```bash
make test
```

## Project Structure

```
flowguard/
├── configs/          # YAML experiment configs
├── data/             # Raw, processed, and split data
├── src/              # Source code (data, models, training, evaluation)
├── scripts/          # CLI entry points
├── notebooks/        # Google Colab notebooks
├── checkpoints/      # Saved model weights
├── results/          # Metrics, plots, logs
└── tests/            # Unit and integration tests
```
