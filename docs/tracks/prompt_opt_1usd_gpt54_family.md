# Prompt-opt track

Single file to change:

```text
src/nanocoop/baselines/prompt_opt.py
```

Run:

```bash
./scripts/run_overcooked_prompt_opt_qwen35_4b_gpt54_budget.sh
```

## Budget

- optimizer spend target: `$1`
- optimizer family target: GPT-5.4 family
- policy target: `Qwen/Qwen3.5-4B`

## What counts as progress

Find a prompt and few-shot package that improves held-out `cross_play_mean_reward` under the same spend cap.

## Typical improvements

- better candidate generation
- smarter pruning
- partner-diverse probe sets
- robust prompt clauses for hidden-info sharing and partner adaptation
