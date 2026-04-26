# Task: OvercookedV2

NanoCoop targets **cooperative post-training** in the **OvercookedV2** setting.

## Core idea

A focal policy must coordinate with **unknown partners** under:

- partial observability
- hidden information
- stochastic transitions
- partner-specific conventions

The main evaluation is **cross-play**, not self-play.

## Backend

The benchmark target is the official **OvercookedV2** stack via **JaxMARL** when available.

Use this for real benchmark runs.

## Starter baseline

The official submission surface is `submission/agent.py`, which must expose
`define()`, `train(data_dir, out_dir)`, and
`eval(checkpoint_dir, data_dir, out_dir)`.

The first no-change focal policy is `gpt-4.1-nano` with the seed cooperative
prompt translated from `configs/starter_agent_gpt41_nano_overcooked_v2.yaml`
into `submission/agent.py`. The model chooses short cooperative macro-action
plans and the adapter executes them for several primitive environment ticks
before replanning.

Run:

```bash
./scripts/run_starter_agent_gpt41_nano.sh
```

Requires `OPENAI_API_KEY` in the environment.

That score is the baseline-to-beat before any offline, RLVR, or
prompt-optimization method changes. NanoCoop is meant to measure cooperative-RL
and post-training method progress, not larger model substitution.

The Qwen3.5 configs remain in-tree as the forward open-model target, matching the NanoHorizon pattern once open serving is ready.

## Official v0.1 evaluation

Official v0.1 records use the pinned default 20 cross-play episodes selected in config. The 48-episode grid remains available for diagnostics and future expansion.

The same `eval()` entrypoint must work both on public train episode IDs and on
held-out leaderboard episode IDs. Only `data_dir` changes.

Run official records with:

```bash
NANOCOOP_TIMEOUT_SECONDS=0 ./scripts/run_starter_agent_gpt41_nano.sh --no-self-play --workers 4 --no-gif
```

The default 180-second script timeout is a developer guard. Timeout-marked records are useful for debugging but are not benchmark-eligible.

## Score

Primary score:

```text
cross_play_mean_reward
```

Reported diagnostics:

- `self_play_mean_reward`
- `mean_completion_rate`
- `cross_partner_std`
- `num_eval_episodes`
- per-layout and per-partner breakdowns
- failed episode IDs, partners, layouts, seeds, rewards, and step counts

## Policy contract

Policies receive a symbolic observation and emit one action from:

- `FETCH_INGREDIENT`
- `PREP_POT`
- `FETCH_DISH`
- `PLATE_SOUP`
- `SERVE_SOUP`
- `WAIT`

The JaxMARL adapter translates this benchmark-level cooking action set into primitive OvercookedV2 actions. The initial benchmark does not expose explicit communication actions because OvercookedV2 has no corresponding primitive communication move in this adapter.

The benchmark package format stays the same across smoke and full-size runs.

## Observation contract

The observation includes:

- layout name
- step index
- shared task progress flags
- the agent's private hint(s)
- focal and partner positions
- focal inventory
- nearby interactables
- pot contents and timers
- loose object summary
- last partner action
- available actions
- compact recent events

That makes the task friendlier for LLM prompting while still keeping hidden information and partner adaptation in play.

## Layout split

Default train layouts:

- `grounded_coord_simple`
- `grounded_coord_ring`
- `demo_cook_simple`

Default held-out eval layouts:

- `test_time_simple`
- `test_time_wide`
- `demo_cook_wide`

You can change the split in config for experiments, but official submissions should pin their split in the record bundle.

## Track philosophy

NanoCoop is not about sweeping config values. It is about improving the **training algorithm**:

- better trace selection
- better curricula
- better partner balancing
- better return weighting
- better protocol adaptation
- better prompt search

That is the edit surface.

Config sweeps, model upgrades, layout edits, and seed edits should be marked experimental / non-eligible unless a track explicitly allows them.
