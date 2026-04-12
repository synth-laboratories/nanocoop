# Records

Each benchmark-eligible run should write a bundle under:

```text
records/<track>/<YYYY-MM-DD>_<run_name>/
```

Minimum expected files:

- `metrics.json`
- `summary.md`
- `policy_package.json`
- `config.yaml`
- `episode_results.jsonl`

Recommended fields in `metrics.json`:

- `primary_score`
- `cross_play_mean_reward`
- `self_play_mean_reward`
- `mean_completion_rate`
- `cross_partner_std`
- `num_eval_episodes`
- `benchmark_eligible`
- `backend`
- `track`
- `run_name`

Do not commit placeholder records. Record bundles should come from verified OvercookedV2 runs.
