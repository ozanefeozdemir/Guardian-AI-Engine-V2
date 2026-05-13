"""V1: Verify FlowGuardProvider uses canavar-model preprocess as a single code path.

The test constructs a raw NF-v3 row, runs it through:
  (a) the provider's _build_vector()
  (b) canavar-model preprocess helpers invoked manually in the training order

and asserts the resulting feature vectors are byte-identical. If the provider
ever drifts from the training pipeline (skips a step, rewrites a transform,
hard-codes a feature list), this test fails.

The test also runs predict() end-to-end to confirm the model returns the
expected dict shape and the argmax decision matches a manual recomputation.

Run:
    python backend/test_flowguard_parity.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
CANAVAR_SRC = os.path.join(PROJECT_ROOT, "model", "canavar-model")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if CANAVAR_SRC not in sys.path:
    sys.path.insert(0, CANAVAR_SRC)

from model_provider import FlowGuardProvider  # noqa: E402
from src.data.features import NF_V3_RAW_FEATURES  # noqa: E402
from src.data.preprocess import (  # noqa: E402
    PreprocessingStats,
    _bucket_ports,
    _handle_inf_and_nan,
    _log_transform,
    _zscore_normalize,
    _IDENTITY_COLUMNS,
    _TIMESTAMP_COLUMNS,
)


def _synthetic_raw_row(seed: int) -> dict:
    """Build a raw NF-v3 dict with realistic ranges keyed on training schema."""
    rng = random.Random(seed)
    row = {}
    # Identity / timestamp (will be dropped by preprocess)
    row["IPV4_SRC_ADDR"] = f"10.0.0.{rng.randint(1, 254)}"
    row["IPV4_DST_ADDR"] = f"10.0.1.{rng.randint(1, 254)}"
    row["L4_SRC_PORT"] = rng.choice([443, 22, 53, 80, 50001, 33445, 8080])
    row["L4_DST_PORT"] = rng.choice([443, 22, 53, 80, 50001, 33445, 8080])
    row["FLOW_START_MILLISECONDS"] = int(rng.random() * 1e12)
    row["FLOW_END_MILLISECONDS"] = row["FLOW_START_MILLISECONDS"] + rng.randint(1, 50000)

    # Numeric features — sample plausibly per heavy-tailed feature.
    heavy = {
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "DNS_TTL_ANSWER",
        "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
        "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
        "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
        "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
    }
    for name in NF_V3_RAW_FEATURES:
        if name in row:
            continue
        if name in heavy:
            row[name] = float(rng.expovariate(1 / 5000.0))
        elif name == "PROTOCOL":
            row[name] = rng.choice([6, 17, 1])  # TCP / UDP / ICMP
        elif name == "L7_PROTO":
            row[name] = rng.randint(0, 250)
        elif name in {"TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"}:
            row[name] = rng.randint(0, 63)
        elif name in {"MIN_TTL", "MAX_TTL"}:
            row[name] = rng.randint(32, 128)
        elif name in {"LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
                       "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN"}:
            row[name] = rng.randint(40, 1514)
        elif name.startswith("NUM_PKTS_"):
            row[name] = rng.randint(0, 200)
        elif name in {"TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"}:
            row[name] = rng.randint(0, 65535)
        elif name == "ICMP_TYPE":
            row[name] = rng.randint(0, 65535)  # combined type*256+code
        elif name == "ICMP_IPV4_TYPE":
            row[name] = rng.randint(0, 255)
        elif name in {"DNS_QUERY_ID", "DNS_QUERY_TYPE"}:
            row[name] = rng.randint(0, 65535)
        elif name == "FTP_COMMAND_RET_CODE":
            row[name] = rng.choice([0, 200, 220, 230, 500])
        else:
            row[name] = 0.0
    return row


def _reference_vector(raw_row: dict, stats: PreprocessingStats) -> np.ndarray:
    """Reproduce preprocess_dataset's transform-mode output for one row.

    Mirrors src.data.preprocess.preprocess_dataset (lines 329-405) but
    invokes the helpers directly rather than going through CSV chunking.
    """
    df = pd.DataFrame([raw_row])
    df.columns = df.columns.str.strip()
    if "L4_SRC_PORT" in df.columns:
        df = pd.concat([df, _bucket_ports(df["L4_SRC_PORT"], "SRC_PORT")], axis=1)
    if "L4_DST_PORT" in df.columns:
        df = pd.concat([df, _bucket_ports(df["L4_DST_PORT"], "DST_PORT")], axis=1)
    cols_to_drop = [c for c in _IDENTITY_COLUMNS + _TIMESTAMP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = _handle_inf_and_nan(df)

    feature_cols = stats.feature_names
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    _log_transform(df, stats.log_transform_columns)
    df, _, _ = _zscore_normalize(
        df,
        feature_cols,
        means=stats.feature_means,
        stds=stats.feature_stds,
    )
    return df[feature_cols].values.astype(np.float32)


def test_parity():
    provider = FlowGuardProvider()
    provider.load()

    n_rows = 100
    max_abs_diff = 0.0
    for seed in range(n_rows):
        raw = _synthetic_raw_row(seed)
        provider_vec = provider._build_vector(raw)
        reference_vec = _reference_vector(raw, provider.stats)

        assert provider_vec.shape == reference_vec.shape, (
            f"Shape mismatch on row {seed}: "
            f"{provider_vec.shape} vs {reference_vec.shape}"
        )
        diff = np.abs(provider_vec - reference_vec).max()
        max_abs_diff = max(max_abs_diff, float(diff))
        if not np.allclose(provider_vec, reference_vec, rtol=1e-6, atol=1e-7):
            mismatched_idx = np.where(
                ~np.isclose(provider_vec, reference_vec, rtol=1e-6, atol=1e-7)
            )[1]
            names = [provider.stats.feature_names[i] for i in mismatched_idx[:10]]
            raise AssertionError(
                f"Row {seed}: vector mismatch. "
                f"max|diff|={diff:.3e}. First mismatched features: {names}"
            )
    print(f"[parity] OK — {n_rows} rows, max|provider − reference| = {max_abs_diff:.3e}")


def test_predict_argmax_consistency():
    provider = FlowGuardProvider()
    provider.load()

    for seed in range(20):
        raw = _synthetic_raw_row(seed)
        vec = provider._build_vector(raw)
        tensor = torch.from_numpy(vec).to(provider.device)
        with torch.no_grad():
            logits = provider.model(tensor)
        manual_pred = int(logits.argmax(dim=1).item())
        manual_p_attack = float(torch.softmax(logits, dim=1).cpu().numpy()[0, 1])

        result = provider.predict(raw)
        assert result["is_attack"] == (manual_pred == 1), (
            f"Row {seed}: predict.is_attack disagrees with argmax. "
            f"manual_pred={manual_pred}, result={result}"
        )
        assert abs(result["confidence"] - manual_p_attack) < 1e-6, (
            f"Row {seed}: confidence not equal to p_attack softmax: "
            f"{result['confidence']} vs {manual_p_attack}"
        )
    print("[predict] OK — argmax decision matches; confidence equals p_attack softmax.")


if __name__ == "__main__":
    test_parity()
    test_predict_argmax_consistency()
    print("\nAll FlowGuard parity tests passed.")
