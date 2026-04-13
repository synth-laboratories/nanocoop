# Offline track

Single file to change:

```text
src/nanocoop/baselines/offline_sft.py
```

Run:

```bash
./scripts/run_offline_training_gpt41_nano.sh
```

## Budget

- wall clock: 20 minutes
- hardware target: 1x A100 40GB
- student target: `gpt-4.1-nano`
- future open-model target: `Qwen/Qwen3.5-4B`

## What counts as progress

Improve held-out `cross_play_mean_reward` under the same budget.

Official v0.1 submissions should change only `src/nanocoop/baselines/offline_sft.py`. Config sweeps, model upgrades, layout edits, and seed edits must be marked experimental / non-eligible.

Run benchmark-eligible records with `NANOCOOP_TIMEOUT_SECONDS=0`; the default 180-second script timeout is only for local iteration.

## Typical improvements

- better teacher filtering
- partner-balanced dataset selection
- layout-balanced sampling
- weighting rare but important recovery traces
- deduplicating redundant conventions
