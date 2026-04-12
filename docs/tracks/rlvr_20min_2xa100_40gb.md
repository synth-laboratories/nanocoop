# RLVR track

Single file to change:

```text
src/nanocoop/baselines/rlvr.py
```

Run:

```bash
./scripts/run_overcooked_rlvr_qwen35_4b_2xa100_20min.sh
```

## Budget

- wall clock: 20 minutes
- hardware target: 2x A100 40GB
- model target: `Qwen/Qwen3.5-4B`

## What counts as progress

Improve held-out `cross_play_mean_reward` or reach the same score in less time.

## Typical improvements

- better rollout grouping
- partner-aware reward weighting
- stronger exploration
- better checkpoint selection
- faster recovery from weak bootstrap conventions
