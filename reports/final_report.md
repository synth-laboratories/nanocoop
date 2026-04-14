# NanoCoop Publication Smoke Report

## Context

Task: `nanocoop-publication-smoke`

Goal: make a minimal, reviewable edit in `submission/agent.py`, keep `train()` / `eval()` working, run a lightweight honest eval slice on train episode IDs, and publish the branch.

## Code change

`submission/agent.py` now includes a small `PUBLICATION_SMOKE_NOTE` constant that is threaded into the submission description and seed prompt.

The eval wrapper also now writes the runner's stdout/stderr to temp files instead of piping captured output directly through `subprocess.run`. That change was necessary because the captured-pipe path was brittle in this workspace and caused the submission eval surface to fail even though the underlying `nanocoop starter-agent` command worked.

## Verification

Smoke setup:

- `train()` and `eval()` were exercised from the submission surface
- train episode slice used: `[2, 7]`
- eval command wrote `result.json`, `stdout.log`, and `stderr.log` under the requested output directory

Observed result on the changed submission:

- `primary_score`: `10.0`
- `mean_reward`: `10.0`
- `num_eval_episodes`: `2`
- `mean_completion_rate`: `0.5`
- episode `2`: reward `20.0`, success `True`
- episode `7`: reward `0.0`, success `False`

## Decision

The publication-smoke note itself is neutral on the smoke slice. There is no evidence from this run that it improves coordination quality.

The wrapper fix is beneficial because it restores the submission `eval()` surface to a working state in this environment.

## Caveat

A direct baseline comparison against the checked-in pre-change wrapper was not reliable here because the old captured-pipe path failed before producing a usable result. I did attempt a baseline comparison, but the environment was also unstable when the underlying OpenAI call was reached from the bare runner.

## Handoff

Modified file:

- `submission/agent.py`

Artifact:

- `reports/final_report.md`
