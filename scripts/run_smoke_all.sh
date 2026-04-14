#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT_SECONDS="${NANOCOOP_TIMEOUT_SECONDS:-180}"

run_with_timeout() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(uv run --project "$ROOT" python "$ROOT/scripts/config_output_dir.py" "$config")"
  NANOCOOP_TIMEOUT_RECORD_DIR="$output_dir" python "$ROOT/scripts/with_timeout.py" "$TIMEOUT_SECONDS" -- "$@"
}

run_with_timeout "$ROOT/configs/offline_smoke.yaml" \
  uv run --project "$ROOT" nanocoop offline --config "$ROOT/configs/offline_smoke.yaml"
run_with_timeout "$ROOT/configs/rlvr_smoke.yaml" \
  uv run --project "$ROOT" nanocoop rlvr --config "$ROOT/configs/rlvr_smoke.yaml"
run_with_timeout "$ROOT/configs/prompt_opt_smoke.yaml" \
  uv run --project "$ROOT" nanocoop prompt-opt --config "$ROOT/configs/prompt_opt_smoke.yaml"
