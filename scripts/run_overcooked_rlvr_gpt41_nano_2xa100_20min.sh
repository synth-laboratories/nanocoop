#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${NANOCOOP_RLVR_CONFIG:-$ROOT/configs/rlvr_overcooked_v2_gpt41_nano_2xa100_20min.yaml}"
TIMEOUT_SECONDS="${NANOCOOP_TIMEOUT_SECONDS:-180}"
OUTPUT_DIR="$(uv run --project "$ROOT" python "$ROOT/scripts/config_output_dir.py" "$CONFIG")"

NANOCOOP_TIMEOUT_RECORD_DIR="$OUTPUT_DIR" python "$ROOT/scripts/with_timeout.py" "$TIMEOUT_SECONDS" -- \
  uv run --project "$ROOT" nanocoop rlvr --config "$CONFIG" "$@"
