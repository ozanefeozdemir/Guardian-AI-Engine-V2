"""
FlowGuard preprocessing pipeline (Section 5.2).

Transforms raw NF-v3 CSV files into normalized Parquet files suitable for
training.  NEVER downloads data or modifies anything under data/raw/.

Supported datasets (from https://staff.itee.uq.edu.au/marius/NIDS_datasets/):
    NF-UNSW-NB15-v3, NF-BoT-IoT-v3, NF-ToN-IoT-v3, NF-CICIDS2018-v3
All four use the unified NetFlow-v3 schema (54 columns + 2 labels).

Usage
-----
Programmatic::

    from src.data.preprocess import run_full_preprocessing
    run_full_preprocessing("configs/base.yaml")

CLI::

    python -m src.data.preprocess --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NF-v3 identity / address columns that must be dropped after port bucketing.
_IDENTITY_COLUMNS = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
]

# Timestamp columns present in v3 data — dropped entirely (absolute time).
_TIMESTAMP_COLUMNS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
]

# Columns eligible for log-transform (heavy-tailed byte / packet counts).
_LOG_TRANSFORM_CANDIDATES = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TOTAL_BYTES",
    "TOTAL_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

# Port-bucket boundaries.
_PORT_BUCKETS = {
    "WELL_KNOWN": (0, 1023),
    "REGISTERED": (1024, 49151),
    "EPHEMERAL": (49152, 65535),
}

# Default NF-v3 label columns.
_LABEL_COL = "Label"
_ATTACK_COL = "Attack"

# Chunk size for reading large CSVs (keeps peak memory reasonable even for
# the 17M-row BoT-IoT and 20M-row CIC datasets).
_CHUNK_SIZE: int = 500_000

# NF-v3 dataset catalog: {short_name: raw_filename}.  The pipeline checks
# for both `.csv` and `.csv.gz` variants.
# Actual files placed by user in data/raw/:
#   NF-UNSW-NB15-v3.csv, NF-BoT-IoT-v3.csv, NF-ToN-IoT-v3.csv,
#   NF-CICIDS2018-v3.csv  (note: CIC variant, not CSE-CIC)
DATASET_CATALOG: dict[str, str] = {
    "unsw": "NF-UNSW-NB15-v3.csv",
    "bot": "NF-BoT-IoT-v3.csv",
    "ton": "NF-ToN-IoT-v3.csv",
    "cic": "NF-CICIDS2018-v3.csv",
}

# URL users should visit to obtain the raw files.
_DOWNLOAD_URL = "https://staff.itee.uq.edu.au/marius/NIDS_datasets/"


# ---------------------------------------------------------------------------
# PreprocessingStats
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingStats:
    """Statistics computed from training data.  Saved alongside model
    checkpoints so that the same normalisation can be applied at inference
    time."""

    feature_means: np.ndarray
    feature_stds: np.ndarray
    log_transform_columns: list[str]
    port_bucket_map: dict
    feature_names: list[str]

    def save(self, path: str) -> None:
        """Persist stats to an ``.npz`` file."""
        np.savez(
            path,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds,
            log_transform_columns=np.array(self.log_transform_columns),
            feature_names=np.array(self.feature_names),
            port_bucket_map_json=np.array(json.dumps(self.port_bucket_map)),
        )
        logger.info("Saved preprocessing stats to %s", path)

    @classmethod
    def load(cls, path: str) -> PreprocessingStats:
        """Load stats from a previously saved ``.npz`` file."""
        data = np.load(path, allow_pickle=True)
        port_map: dict = {}
        if "port_bucket_map_json" in data:
            port_map = json.loads(str(data["port_bucket_map_json"]))
        return cls(
            feature_means=data["feature_means"],
            feature_stds=data["feature_stds"],
            log_transform_columns=data["log_transform_columns"].tolist(),
            port_bucket_map=port_map,
            feature_names=data["feature_names"].tolist(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_raw_path(raw_dir: str, filename: str) -> pathlib.Path:
    """Return the path to a raw CSV, accepting both ``.csv`` and ``.csv.gz``.

    Raises ``FileNotFoundError`` with an actionable message when neither
    variant exists.
    """
    csv_path = pathlib.Path(raw_dir) / filename
    gz_path = pathlib.Path(raw_dir) / (filename + ".gz")

    if csv_path.exists():
        return csv_path
    if gz_path.exists():
        return gz_path

    raise FileNotFoundError(
        f"Raw data file not found.  Please place '{filename}' (or "
        f"'{filename}.gz') in '{raw_dir}'.  "
        f"You can obtain it from: {_DOWNLOAD_URL}"
    )


def _bucket_ports(ports: pd.Series, prefix: str) -> pd.DataFrame:
    """Create one-hot port-bucket features from a port series.

    Parameters
    ----------
    ports : pd.Series
        Integer port numbers.
    prefix : str
        ``"SRC_PORT"`` or ``"DST_PORT"``.

    Returns
    -------
    pd.DataFrame
        Three boolean columns per prefix (WELL_KNOWN, REGISTERED, EPHEMERAL).
    """
    ports = pd.to_numeric(ports, errors="coerce").fillna(0).astype(np.int64)
    result = pd.DataFrame(index=ports.index)
    for bucket_name, (lo, hi) in _PORT_BUCKETS.items():
        col = f"{prefix}_{bucket_name}"
        result[col] = ((ports >= lo) & (ports <= hi)).astype(np.int8)
    return result


def _handle_inf_and_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities with per-column finite max, then fill NaN with 0."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col]
        inf_mask = np.isinf(series)
        if inf_mask.any():
            finite_vals = series[np.isfinite(series)]
            clip_val = finite_vals.max() if len(finite_vals) > 0 else 0.0
            df[col] = series.where(~inf_mask, clip_val)
    df = df.fillna(0)
    return df


def _log_transform(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Apply ``log1p(|x|)`` to *existing* heavy-tailed columns in-place.

    Returns the list of columns that were actually transformed.
    """
    transformed: list[str] = []
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(np.abs(df[col].astype(np.float64)))
            transformed.append(col)
    return transformed


def _zscore_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Z-score normalise the given feature columns.

    When *means* and *stds* are ``None`` the statistics are computed from
    ``df`` (fit mode).  Otherwise the supplied statistics are applied
    (transform mode).

    Returns ``(df, means, stds)``.
    """
    subset = df[feature_cols].astype(np.float64)
    if means is None or stds is None:
        means = subset.mean().values
        stds = subset.std().values
    df[feature_cols] = (subset.values - means) / (stds + eps)
    return df, means, stds


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def preprocess_dataset(
    raw_path: str,
    output_path: str,
    *,
    label_map: dict[str, int] | None = None,
    fit_stats: bool = True,
    stats: PreprocessingStats | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> PreprocessingStats:
    """Preprocess a single NF-v3 CSV dataset into a Parquet file.

    Parameters
    ----------
    raw_path:
        Path to the raw ``.csv`` or ``.csv.gz`` file.
    output_path:
        Destination Parquet file.
    label_map:
        Optional mapping from attack-type strings to unified integer codes.
        If ``None``, labels are kept as-is.
    fit_stats:
        If ``True`` (default), compute normalisation statistics from this
        dataset.  If ``False``, *stats* must be provided.
    stats:
        A :class:`PreprocessingStats` instance to use when
        ``fit_stats=False``.
    chunk_size:
        Number of rows per read chunk.  Keeps memory bounded for
        multi-million-row files.

    Returns
    -------
    PreprocessingStats
        The statistics used (either freshly computed or the ones passed in).
    """
    raw_path_resolved = pathlib.Path(raw_path)
    if not raw_path_resolved.exists():
        raise FileNotFoundError(
            f"Raw data file not found: '{raw_path}'.  "
            f"You can obtain it from: {_DOWNLOAD_URL}"
        )

    if not fit_stats and stats is None:
        raise ValueError(
            "fit_stats=False requires a PreprocessingStats instance via the "
            "'stats' parameter."
        )

    logger.info("Reading %s in chunks of %d rows ...", raw_path, chunk_size)

    # ------------------------------------------------------------------
    # Pass 1 – load + transform in chunks, accumulate into a list
    # ------------------------------------------------------------------
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        raw_path,
        low_memory=False,
        chunksize=chunk_size,
    )

    for i, chunk in enumerate(reader):
        # Strip any whitespace from column names (some CSVs have trailing spaces).
        chunk.columns = chunk.columns.str.strip()

        # --- Port bucketing (before dropping identity columns) -----
        if "L4_SRC_PORT" in chunk.columns:
            src_buckets = _bucket_ports(chunk["L4_SRC_PORT"], "SRC_PORT")
            chunk = pd.concat([chunk, src_buckets], axis=1)
        if "L4_DST_PORT" in chunk.columns:
            dst_buckets = _bucket_ports(chunk["L4_DST_PORT"], "DST_PORT")
            chunk = pd.concat([chunk, dst_buckets], axis=1)

        # --- Drop identity and timestamp columns ------------------
        cols_to_drop = [c for c in _IDENTITY_COLUMNS + _TIMESTAMP_COLUMNS if c in chunk.columns]
        chunk = chunk.drop(columns=cols_to_drop)

        # --- Map attack labels to unified taxonomy -----------------
        if label_map and _ATTACK_COL in chunk.columns:
            chunk[_ATTACK_COL] = chunk[_ATTACK_COL].map(
                lambda x: label_map.get(str(x).strip(), label_map.get("__default__", x))
            )

        # --- Inf / NaN handling (per-chunk) -------------------------
        chunk = _handle_inf_and_nan(chunk)

        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            logger.info(
                "  ... processed %d chunks (%d rows so far)",
                i + 1,
                (i + 1) * chunk_size,
            )

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))
    del chunks  # free memory

    # ------------------------------------------------------------------
    # Identify feature columns (everything except label columns)
    # ------------------------------------------------------------------
    label_cols_present = [c for c in (_LABEL_COL, _ATTACK_COL) if c in df.columns]
    feature_cols = [c for c in df.columns if c not in label_cols_present]

    # ------------------------------------------------------------------
    # Log-transform heavy-tailed features
    # ------------------------------------------------------------------
    actually_transformed = _log_transform(df, _LOG_TRANSFORM_CANDIDATES)
    logger.info("Log-transformed columns: %s", actually_transformed)

    # ------------------------------------------------------------------
    # Z-score normalisation
    # ------------------------------------------------------------------
    if fit_stats:
        df, means, stds = _zscore_normalize(df, feature_cols)
        computed_stats = PreprocessingStats(
            feature_means=means,
            feature_stds=stds,
            log_transform_columns=actually_transformed,
            port_bucket_map=dict(_PORT_BUCKETS),
            feature_names=feature_cols,
        )
    else:
        assert stats is not None
        # Align feature columns to those in the provided stats.  If new
        # columns appear that were not in training, drop them; if training
        # columns are missing, create zero-filled columns.
        for col in stats.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        feature_cols = stats.feature_names
        df, means, stds = _zscore_normalize(
            df,
            feature_cols,
            means=stats.feature_means,
            stds=stats.feature_stds,
        )
        computed_stats = stats

    # ------------------------------------------------------------------
    # Ensure all feature columns are float32 for storage efficiency
    # ------------------------------------------------------------------
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)

    # ------------------------------------------------------------------
    # Save as Parquet
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[feature_cols + label_cols_present].to_parquet(
        output_path, index=False, engine="pyarrow"
    )
    logger.info("Saved %d rows to %s", len(df), output_path)

    return computed_stats


# ---------------------------------------------------------------------------
# Combined unlabeled set (for self-supervised / contrastive pretraining)
# ---------------------------------------------------------------------------


def create_combined_unlabeled(
    processed_dir: str,
    dataset_names: list[str],
    output_filename: str = "combined_unlabeled.parquet",
    seed: int = 42,
) -> pathlib.Path:
    """Merge all benign traffic from the processed Parquet files into a
    single, shuffled, unlabeled Parquet file.

    Parameters
    ----------
    processed_dir:
        Directory containing the per-dataset Parquet files.
    dataset_names:
        List of dataset short names whose processed files should be merged.
    output_filename:
        Name of the output file (written inside *processed_dir*).
    seed:
        Random seed for shuffling.

    Returns
    -------
    pathlib.Path
        Path to the written Parquet file.
    """
    parts: list[pd.DataFrame] = []

    for name in dataset_names:
        parquet_path = pathlib.Path(processed_dir) / f"{name}.parquet"
        if not parquet_path.exists():
            logger.warning(
                "Processed file %s not found – skipping for combined set.",
                parquet_path,
            )
            continue

        df = pd.read_parquet(parquet_path, engine="pyarrow")
        logger.info("Loaded %s: %d rows", parquet_path.name, len(df))

        # Keep only benign traffic (Label == 0).
        if _LABEL_COL in df.columns:
            df = df[df[_LABEL_COL] == 0]
            logger.info("  -> %d benign rows after filtering.", len(df))

        # Drop label columns.
        label_cols = [c for c in (_LABEL_COL, _ATTACK_COL) if c in df.columns]
        df = df.drop(columns=label_cols)

        parts.append(df)

    if not parts:
        raise RuntimeError(
            "No processed datasets found.  Run preprocess_dataset first."
        )

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path = pathlib.Path(processed_dir) / output_filename
    combined.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(
        "Saved combined unlabeled set: %d rows -> %s", len(combined), out_path
    )
    return out_path


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_full_preprocessing(config_path: str) -> dict[str, PreprocessingStats]:
    """Execute the complete preprocessing pipeline driven by a YAML config.

    Steps
    -----
    1. For each dataset in the catalog, locate the raw file and preprocess it.
    2. Build the combined unlabeled Parquet for pretraining.
    3. Print summary statistics.

    Parameters
    ----------
    config_path:
        Path to the base YAML config (e.g., ``configs/base.yaml``).

    Returns
    -------
    dict[str, PreprocessingStats]
        Mapping from dataset name to the normalisation statistics computed
        during preprocessing.
    """
    config = load_config(config_path)
    paths = config.get("paths", {})
    raw_dir = paths.get("raw_data", "data/raw")
    processed_dir = paths.get("processed_data", "data/processed")
    seed = config.get("project", {}).get("seed", 42)

    # Optional per-attack label mapping from config.
    label_map: dict[str, int] | None = config.get("preprocessing", {}).get(
        "label_map", None
    )

    all_stats: dict[str, PreprocessingStats] = {}
    dataset_names: list[str] = []

    for name, filename in DATASET_CATALOG.items():
        logger.info("=" * 60)
        logger.info("Processing dataset: %s", name)
        logger.info("=" * 60)

        try:
            raw_path = _resolve_raw_path(raw_dir, filename)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            logger.warning("Skipping %s.", name)
            continue

        output_path = os.path.join(processed_dir, f"{name}.parquet")
        stats = preprocess_dataset(
            raw_path=str(raw_path),
            output_path=output_path,
            label_map=label_map,
            fit_stats=True,
        )
        all_stats[name] = stats
        dataset_names.append(name)

        # Persist stats alongside processed data.
        stats_path = os.path.join(processed_dir, f"{name}_stats.npz")
        stats.save(stats_path)

    # ------------------------------------------------------------------
    # Combined unlabeled set
    # ------------------------------------------------------------------
    if dataset_names:
        logger.info("=" * 60)
        logger.info("Creating combined unlabeled set ...")
        logger.info("=" * 60)
        create_combined_unlabeled(
            processed_dir=processed_dir,
            dataset_names=dataset_names,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    for name, st in all_stats.items():
        print(f"  {name}:")
        print(f"    Features        : {len(st.feature_names)}")
        print(f"    Log-transformed : {st.log_transform_columns}")
        print(f"    Mean range      : [{st.feature_means.min():.4f}, {st.feature_means.max():.4f}]")
        print(f"    Std range       : [{st.feature_stds.min():.4f}, {st.feature_stds.max():.4f}]")
    if not all_stats:
        print("  (no datasets were processed – raw files may be missing)")
    print("=" * 60 + "\n")

    return all_stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="FlowGuard preprocessing pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    run_full_preprocessing(args.config)
