#!/usr/bin/env python3
"""Validate that raw dataset files are present and well-formed."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.validate_raw import validate_raw_datasets


def main():
    report = validate_raw_datasets(config_path="configs/base.yaml")

    all_ok = all(info["exists"] for info in report.values())
    if not all_ok:
        print("\nMISSING DATASETS. Place the following files in data/raw/ before continuing:")
        for ds_name, info in report.items():
            if not info["exists"]:
                print(f"   -> {info['expected_file']}")
        print("\n   Download from: https://staff.itee.uq.edu.au/marius/NIDS_datasets/")
        sys.exit(1)
    else:
        print("\nAll datasets found and validated.")
        for ds_name, info in report.items():
            print(f"   {ds_name}: {info['rows']:,} rows, {info['cols']} columns, {info['size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
