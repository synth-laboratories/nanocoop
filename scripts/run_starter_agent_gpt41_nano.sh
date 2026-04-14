#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${NANOCOOP_STARTER_CONFIG:-$ROOT/configs/starter_agent_gpt41_nano_overcooked_v2.yaml}"
TIMEOUT_SECONDS="${NANOCOOP_TIMEOUT_SECONDS:-180}"
OUTPUT_DIR="$(uv run --project "$ROOT" python "$ROOT/scripts/config_output_dir.py" "$CONFIG")"

NANOCOOP_TIMEOUT_RECORD_DIR="$OUTPUT_DIR" python "$ROOT/scripts/with_timeout.py" "$TIMEOUT_SECONDS" -- \
  uv run --project "$ROOT" nanocoop starter-agent --config "$CONFIG" "$@"
