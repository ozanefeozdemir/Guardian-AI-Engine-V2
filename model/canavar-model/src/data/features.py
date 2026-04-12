"""
Feature schema for NF-v3 network flow datasets.

This module is the single source of truth for all feature definitions used
throughout the FlowGuard pipeline. Features are organized into semantic
categories that reflect their role in network traffic analysis.

Actual column header from NF-v3 CSV files:
    FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS, IPV4_SRC_ADDR,
    L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT, PROTOCOL, L7_PROTO,
    IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS, TCP_FLAGS, CLIENT_TCP_FLAGS,
    SERVER_TCP_FLAGS, FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT,
    MIN_TTL, MAX_TTL, LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN,
    MAX_IP_PKT_LEN, SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES,
    RETRANSMITTED_IN_BYTES, RETRANSMITTED_IN_PKTS, RETRANSMITTED_OUT_BYTES,
    RETRANSMITTED_OUT_PKTS, SRC_TO_DST_AVG_THROUGHPUT,
    DST_TO_SRC_AVG_THROUGHPUT, NUM_PKTS_UP_TO_128_BYTES,
    NUM_PKTS_128_TO_256_BYTES, NUM_PKTS_256_TO_512_BYTES,
    NUM_PKTS_512_TO_1024_BYTES, NUM_PKTS_1024_TO_1514_BYTES,
    TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT, ICMP_TYPE, ICMP_IPV4_TYPE,
    DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE,
    SRC_TO_DST_IAT_MIN, SRC_TO_DST_IAT_MAX, SRC_TO_DST_IAT_AVG,
    SRC_TO_DST_IAT_STDDEV, DST_TO_SRC_IAT_MIN, DST_TO_SRC_IAT_MAX,
    DST_TO_SRC_IAT_AVG, DST_TO_SRC_IAT_STDDEV, Label, Attack
"""

from typing import List

# ---------------------------------------------------------------------------
# Timestamp columns -- dropped during preprocessing (encode absolute time)
# ---------------------------------------------------------------------------
TIMESTAMP_FEATURES: List[str] = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
]

# ---------------------------------------------------------------------------
# Identity features -- used for port bucketing then dropped
# ---------------------------------------------------------------------------
IDENTITY_FEATURES: List[str] = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
]

# ---------------------------------------------------------------------------
# Protocol features -- transport / application layer identifiers
# ---------------------------------------------------------------------------
PROTOCOL_FEATURES: List[str] = [
    "PROTOCOL",
    "L7_PROTO",
]

# ---------------------------------------------------------------------------
# Volume features -- byte and packet counts
# ---------------------------------------------------------------------------
VOLUME_FEATURES: List[str] = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
]

# ---------------------------------------------------------------------------
# Timing / duration features
# ---------------------------------------------------------------------------
TIMING_FEATURES: List[str] = [
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
]

# ---------------------------------------------------------------------------
# Inter-arrival time (IAT) features -- v3 specific
# ---------------------------------------------------------------------------
IAT_FEATURES: List[str] = [
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

# ---------------------------------------------------------------------------
# Ratio / throughput features
# ---------------------------------------------------------------------------
RATIO_FEATURES: List[str] = [
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
]

# ---------------------------------------------------------------------------
# TCP flag features
# ---------------------------------------------------------------------------
TCP_FLAG_FEATURES: List[str] = [
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
]

# ---------------------------------------------------------------------------
# IP / TTL features
# ---------------------------------------------------------------------------
TTL_FEATURES: List[str] = [
    "MIN_TTL",
    "MAX_TTL",
]

# ---------------------------------------------------------------------------
# Packet length features
# ---------------------------------------------------------------------------
PACKET_LENGTH_FEATURES: List[str] = [
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
]

# ---------------------------------------------------------------------------
# Retransmission features
# ---------------------------------------------------------------------------
RETRANSMISSION_FEATURES: List[str] = [
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
]

# ---------------------------------------------------------------------------
# Packet-size distribution features (histogram buckets)
# ---------------------------------------------------------------------------
PACKET_SIZE_DISTRIBUTION_FEATURES: List[str] = [
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
]

# ---------------------------------------------------------------------------
# TCP window features
# ---------------------------------------------------------------------------
TCP_WINDOW_FEATURES: List[str] = [
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
]

# ---------------------------------------------------------------------------
# ICMP features
# ---------------------------------------------------------------------------
ICMP_FEATURES: List[str] = [
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
]

# ---------------------------------------------------------------------------
# DNS features
# ---------------------------------------------------------------------------
DNS_FEATURES: List[str] = [
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
]

# ---------------------------------------------------------------------------
# FTP features
# ---------------------------------------------------------------------------
FTP_FEATURES: List[str] = [
    "FTP_COMMAND_RET_CODE",
]

# ---------------------------------------------------------------------------
# Port bucket features -- computed during preprocessing from raw ports
# ---------------------------------------------------------------------------
PORT_BUCKET_FEATURES: List[str] = [
    "SRC_PORT_WELL_KNOWN",
    "SRC_PORT_REGISTERED",
    "SRC_PORT_EPHEMERAL",
    "DST_PORT_WELL_KNOWN",
    "DST_PORT_REGISTERED",
    "DST_PORT_EPHEMERAL",
]

# ---------------------------------------------------------------------------
# Engineered features -- computed during preprocessing
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES: List[str] = PORT_BUCKET_FEATURES

# ---------------------------------------------------------------------------
# Label columns
# ---------------------------------------------------------------------------
LABEL_COLUMNS: List[str] = [
    "Label",    # binary: 0 = benign, 1 = malicious
    "Attack",   # multi-class attack type string / category
]

# ---------------------------------------------------------------------------
# Features that benefit from log1p transformation due to heavy skew
# ---------------------------------------------------------------------------
LOG_TRANSFORM_FEATURES: List[str] = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
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

# ---------------------------------------------------------------------------
# All raw NF-v3 columns (as they appear in the CSV header)
# ---------------------------------------------------------------------------
NF_V3_RAW_FEATURES: List[str] = (
    TIMESTAMP_FEATURES
    + IDENTITY_FEATURES
    + PROTOCOL_FEATURES
    + VOLUME_FEATURES
    + TCP_FLAG_FEATURES
    + TIMING_FEATURES
    + TTL_FEATURES
    + PACKET_LENGTH_FEATURES
    + RATIO_FEATURES
    + RETRANSMISSION_FEATURES
    + PACKET_SIZE_DISTRIBUTION_FEATURES
    + TCP_WINDOW_FEATURES
    + ICMP_FEATURES
    + DNS_FEATURES
    + FTP_FEATURES
    + IAT_FEATURES
)

NF_V3_EXPECTED_COLUMNS: List[str] = NF_V3_RAW_FEATURES + LABEL_COLUMNS

# Columns to drop during preprocessing (identity + timestamps)
DROP_COLUMNS: List[str] = IDENTITY_FEATURES + TIMESTAMP_FEATURES

# ---------------------------------------------------------------------------
# All model-input features (raw numeric + engineered, excluding dropped/labels)
# ---------------------------------------------------------------------------
_MODEL_INPUT_RAW_FEATURES: List[str] = (
    PROTOCOL_FEATURES
    + VOLUME_FEATURES
    + TCP_FLAG_FEATURES
    + TIMING_FEATURES
    + TTL_FEATURES
    + PACKET_LENGTH_FEATURES
    + RATIO_FEATURES
    + RETRANSMISSION_FEATURES
    + PACKET_SIZE_DISTRIBUTION_FEATURES
    + TCP_WINDOW_FEATURES
    + ICMP_FEATURES
    + DNS_FEATURES
    + FTP_FEATURES
    + IAT_FEATURES
)

# Forward/backward swap pairs for augmentation
SWAP_PAIRS: List[tuple] = [
    ("IN_BYTES", "OUT_BYTES"),
    ("IN_PKTS", "OUT_PKTS"),
    ("SRC_TO_DST_IAT_MIN", "DST_TO_SRC_IAT_MIN"),
    ("SRC_TO_DST_IAT_MAX", "DST_TO_SRC_IAT_MAX"),
    ("SRC_TO_DST_IAT_AVG", "DST_TO_SRC_IAT_AVG"),
    ("SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_STDDEV"),
    ("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"),
    ("SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"),
    ("CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"),
    ("TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"),
    ("DURATION_IN", "DURATION_OUT"),
    ("RETRANSMITTED_IN_BYTES", "RETRANSMITTED_OUT_BYTES"),
    ("RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS"),
]


def get_feature_names(include_engineered: bool = True) -> List[str]:
    """Return the ordered list of feature names used as model inputs.

    Includes all raw numeric features (excluding identity, timestamp, and
    label columns) plus optionally the port bucket features.
    """
    features = list(_MODEL_INPUT_RAW_FEATURES)
    if include_engineered:
        features = features + ENGINEERED_FEATURES
    return features


def get_input_dim(include_engineered: bool = True) -> int:
    """Return the total number of input features for the model."""
    return len(get_feature_names(include_engineered=include_engineered))
