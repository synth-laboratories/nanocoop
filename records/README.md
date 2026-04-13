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
- `layout_breakdown`
- `partner_breakdown`
- `failed_episodes`
- `official_episode_ids`
- `timeout_seconds`
- `timed_out`
- `official_record`
- `benchmark_eligible`
- `backend`
- `track`
- `run_name`

Official v0.1 records must run with `NANOCOOP_TIMEOUT_SECONDS=0`. Developer-timeout runs may leave `timeout_status.json`; those are not benchmark-eligible.

Do not commit placeholder records. Record bundles should come from verified OvercookedV2 runs.
