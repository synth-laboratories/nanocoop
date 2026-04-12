# Offline track

Single file to change:

```text
src/nanocoop/baselines/offline_sft.py
```

Run:

```bash
./scripts/run_offline_training.sh
```

## Budget

- wall clock: 20 minutes
- hardware target: 1x A100 40GB
- student target: `Qwen/Qwen3.5-4B`
- teacher target: `Qwen/Qwen3.5-9B`

## What counts as progress

Improve held-out `cross_play_mean_reward` under the same budget.

## Typical improvements

- better teacher filtering
- partner-balanced dataset selection
- layout-balanced sampling
- weighting rare but important recovery traces
- deduplicating redundant conventions
