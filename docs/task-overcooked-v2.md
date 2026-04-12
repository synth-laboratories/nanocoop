# Task: OvercookedV2

NanoCoop targets **cooperative post-training** in the **OvercookedV2** setting.

## Core idea

A focal policy must coordinate with **unknown partners** under:

- partial observability
- hidden information
- stochastic transitions
- partner-specific conventions

The main evaluation is **cross-play**, not self-play.

## Backend modes

### 1. Official backend

The benchmark target is the official **OvercookedV2** stack via **JaxMARL** when available.

Use this for real benchmark runs.

### 2. Smoke backend

This repo ships a local, fully self-contained **mock backend** that preserves the benchmark pressure:

- each side has private information
- partner styles differ
- layouts induce slightly different coordination structure
- stochastic failures require recovery
- cross-play matters more than memorizing one convention

Use this for:

- CI
- algorithm debugging
- fast local iteration
- baseline sanity checks

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

## Policy contract

Policies receive a symbolic observation and emit one action from:

- `SHARE_RECIPE`
- `SHARE_POT`
- `FETCH_INGREDIENT`
- `PREP_POT`
- `FETCH_DISH`
- `PLATE_SOUP`
- `SERVE_SOUP`
- `WAIT`

The smoke backend uses this exact action set. The official backend adapter may expose richer internal state, but the benchmark package format stays the same.

## Observation contract

The observation includes:

- layout name
- step index
- shared task progress flags
- the agent's private hint(s)
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
