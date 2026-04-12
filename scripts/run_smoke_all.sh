#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m nanocoop.cli offline --config "$ROOT/configs/offline_smoke.yaml"
python -m nanocoop.cli rlvr --config "$ROOT/configs/rlvr_smoke.yaml"
python -m nanocoop.cli prompt-opt --config "$ROOT/configs/prompt_opt_smoke.yaml"
