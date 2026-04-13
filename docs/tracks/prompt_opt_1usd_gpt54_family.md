# Prompt-opt track

Single file to change:

```text
src/nanocoop/baselines/prompt_opt.py
```

Run:

```bash
./scripts/run_overcooked_prompt_opt_gpt41_nano_gpt54_budget.sh
```

## Budget

- optimizer spend target: `$1`
- optimizer family target: GPT-5.4 family
- policy target: `gpt-4.1-nano`
- future open-model target: `Qwen/Qwen3.5-4B`

## What counts as progress

Find a prompt and few-shot package that improves held-out `cross_play_mean_reward` under the same spend cap.

Official v0.1 submissions should change only `src/nanocoop/baselines/prompt_opt.py`. Config sweeps, model upgrades, layout edits, and seed edits must be marked experimental / non-eligible.

Run benchmark-eligible records with `NANOCOOP_TIMEOUT_SECONDS=0`; the default 180-second script timeout is only for local iteration.

## Typical improvements

- better candidate generation
- smarter pruning
- partner-diverse probe sets
- robust prompt clauses for hidden-info sharing and partner adaptation
