#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${NANOCOOP_OFFLINE_CONFIG:-$ROOT/configs/offline_overcooked_v2_qwen35_4b.yaml}"

python -m nanocoop.cli offline --config "$CONFIG" "$@"
