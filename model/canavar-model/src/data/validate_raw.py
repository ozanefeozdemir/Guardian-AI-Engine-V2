"""
Raw dataset validation for NF-v3 network flow datasets.

THE USER MANUALLY PLACES RAW CSV FILES INTO data/raw/.
This module does NOT download anything. It only validates that the expected
files exist, checks their structure, and reports basic statistics.

DO NOT write any code that downloads files from the internet.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.features import LABEL_COLUMNS, NF_V3_EXPECTED_COLUMNS
from src.utils.config import load_config

logger = logging.getLogger(__name__)

_DOWNLOAD_URL = "https://staff.itee.uq.edu.au/marius/NIDS_datasets/"


def _count_rows_chunked(file_path: str, chunk_size: int = 100_000) -> int:
    """Count total rows via chunked reading to avoid OOM on large files."""
    total = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        total += len(chunk)
    return total


def _read_header(file_path: str) -> List[str]:
    """Read only the header row of a CSV (or .csv.gz) file."""
    df = pd.read_csv(file_path, nrows=0)
    return list(df.columns.str.strip())


def _get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def check_schema_compatibility(csv_path: str, expected_columns: Optional[List[str]] = None) -> dict:
    """
    Read the header row of a CSV file and verify it contains expected NF-v3 columns.

    Returns:
      - is_compatible: bool
      - expected_columns: list[str]
      - found_columns: list[str]
      - missing_columns: list[str]
      - extra_columns: list[str]
    """
    if expected_columns is None:
        expected_columns = NF_V3_EXPECTED_COLUMNS

    found_columns = _read_header(csv_path)
    expected_set = set(expected_columns)
    found_set = set(found_columns)

    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set)

    return {
        "is_compatible": len(missing) == 0,
        "expected_columns": expected_columns,
        "found_columns": found_columns,
        "missing_columns": missing,
        "extra_columns": extra,
    }


def validate_raw_datasets(
    config_path: str = "configs/base.yaml",
    count_rows: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Check that all expected raw CSV files exist in data/raw/.

    For each dataset returns:
      - exists: bool
      - expected_file: str
      - rows: int (-1 if missing or count_rows=False)
      - cols: int (-1 if missing)
      - size_mb: float (0 if missing)
      - columns_match: bool
      - label_column_present: bool

    Does NOT load entire files into memory.
    If a file is missing, returns exists=False — does NOT raise.
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    datasets = data_cfg.get("datasets", [])

    report = {}

    for ds_info in datasets:
        name = ds_info["name"]
        raw_file = ds_info.get("raw_file", "")

        # Check both .csv and .csv.gz
        candidate_paths = [raw_file, raw_file + ".gz"]
        if not raw_file.endswith(".gz"):
            candidate_paths = [raw_file, raw_file + ".gz"]
        else:
            candidate_paths = [raw_file]

        found_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path is None:
            report[name] = {
                "exists": False,
                "expected_file": raw_file,
                "rows": -1,
                "cols": -1,
                "size_mb": 0.0,
                "columns_match": False,
                "label_column_present": False,
            }
            print(f"Missing: {raw_file} — please download from {_DOWNLOAD_URL} and place it in data/raw/")
            continue

        # File exists — gather stats
        size_mb = _get_file_size_mb(found_path)

        try:
            header = _read_header(found_path)
            cols = len(header)
            label_present = "Label" in header and "Attack" in header
            # Quick schema check against actual v3 header
            schema = check_schema_compatibility(found_path)
            columns_match = schema["is_compatible"]
        except Exception as e:
            logger.warning("Could not read header for %s: %s", found_path, e)
            cols = -1
            label_present = False
            columns_match = False

        rows = -1
        if count_rows:
            try:
                logger.info("Counting rows for %s (may take a moment for large files)...", name)
                rows = _count_rows_chunked(found_path)
            except Exception as e:
                logger.warning("Could not count rows for %s: %s", found_path, e)

        report[name] = {
            "exists": True,
            "expected_file": raw_file,
            "found_file": found_path,
            "rows": rows,
            "cols": cols,
            "size_mb": size_mb,
            "columns_match": columns_match,
            "label_column_present": label_present,
        }

    return report


def print_validation_report(report: Dict[str, Dict[str, Any]]) -> None:
    """Print a human-readable validation report."""
    print("\n" + "=" * 60)
    print("FlowGuard Raw Dataset Validation")
    print("=" * 60)
    for name, info in report.items():
        status = "OK" if info["exists"] else "MISSING"
        print(f"\n[{status}] {name}")
        print(f"  Expected : {info['expected_file']}")
        if info["exists"]:
            print(f"  Found    : {info.get('found_file', info['expected_file'])}")
            print(f"  Size     : {info['size_mb']:.1f} MB")
            if info["rows"] >= 0:
                print(f"  Rows     : {info['rows']:,}")
            print(f"  Columns  : {info['cols']}")
            print(f"  Schema   : {'OK' if info['columns_match'] else 'MISMATCH (non-fatal)'}")
            print(f"  Labels   : {'OK' if info['label_column_present'] else 'MISSING'}")
        else:
            print(f"  -> Download from: {_DOWNLOAD_URL}")
    print("=" * 60 + "\n")
