#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${NANOCOOP_RLVR_CONFIG:-$ROOT/configs/rlvr_overcooked_v2_qwen35_4b_2xa100_20min.yaml}"

python -m nanocoop.cli rlvr --config "$CONFIG" "$@"
