#!/usr/bin/env python3
"""Export immutable, labelled production examples from the optional journal."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from training.export import export_examples

# Backward-compatible function name used by local callers and tests.
export = export_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_examples(args.store, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
