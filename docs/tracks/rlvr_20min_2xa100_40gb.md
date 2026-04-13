# RLVR track

Single file to change:

```text
src/nanocoop/baselines/rlvr.py
```

Run:

```bash
./scripts/run_overcooked_rlvr_gpt41_nano_2xa100_20min.sh
```

## Budget

- wall clock: 20 minutes
- hardware target: 2x A100 40GB
- policy target: `gpt-4.1-nano`
- future open-model target: `Qwen/Qwen3.5-4B`

## What counts as progress

Improve held-out `cross_play_mean_reward` or reach the same score in less time.

Official v0.1 submissions should change only `src/nanocoop/baselines/rlvr.py`. Config sweeps, model upgrades, layout edits, and seed edits must be marked experimental / non-eligible.

Run benchmark-eligible records with `NANOCOOP_TIMEOUT_SECONDS=0`; the default 180-second script timeout is only for local iteration.

## Typical improvements

- better rollout grouping
- partner-aware reward weighting
- stronger exploration
- better checkpoint selection
- faster recovery from weak bootstrap conventions
