#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    path = Path(sys.argv[1])
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    output_dir = data.get("output_dir", "")
    if output_dir:
        print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
