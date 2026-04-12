#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${NANOCOOP_PROMPT_OPT_CONFIG:-$ROOT/configs/prompt_opt_overcooked_v2_qwen35_4b_gpt54_budget.yaml}"

python -m nanocoop.cli prompt-opt --config "$CONFIG" "$@"
