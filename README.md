# NanoCoop

**Fast, cheap iteration for cooperative LLM agents.** Improve agents on **OvercookedV2** under hard time, hardware, and budget caps with reproducible runs, pinned metrics, and public records anyone can verify.

NanoCoop mirrors the spirit of NanoHorizon, but swaps the long-horizon single-agent Craftax setting for a **cooperative, partially observable, stochastic coordination task**. Each track gives you **one Python file** containing a baseline algorithm. Your job is to write a better one.

> The benchmark target is the official **OvercookedV2** environment stack via **JaxMARL**.

---

## Leaderboard Submission Surface

For leaderboard and SMR submissions, edit only:

`submission/agent.py`

That file owns the full submission contract:

- `define() -> dict`
- `train(data_dir, out_dir) -> None`
- `eval(checkpoint_dir, data_dir, out_dir) -> dict`

The same `eval()` entrypoint is used for public train episode IDs and held-out
leaderboard episode IDs. The scorer only changes `data_dir`.

`eval()` should emit rollout record artifacts, including per-episode GIFs when
available, so the leaderboard candidate page can attach media for each rollout.

## What this benchmark is about

NanoCoop is about **changing the training algorithm**, not tweaking config knobs.

Each track gives you a single Python file containing a starter method:

- `src/nanocoop/baselines/offline_sft.py`
- `src/nanocoop/baselines/rlvr.py`
- `src/nanocoop/baselines/prompt_opt.py`

Change the algorithm. Run the track script. Check your score.

There are two ways to win:

1. **Higher score** — get more cooperative return against held-out partners and layouts under the same budget.
2. **Higher throughput** — get the same score faster, or more lift per minute / dollar.

Starter policy target for the first public baseline: `gpt-4.1-nano`.

Open-model target for the main training tracks: `Qwen/Qwen3.5-4B`, unless a track doc states otherwise. The Qwen configs stay in-tree so the benchmark can move to NanoHorizon-style open-model runs once serving is ready.

---

## Why OvercookedV2

Classic Overcooked is a useful cooperative benchmark, but OvercookedV2 adds the ingredients NanoCoop actually wants to stress:

- **asymmetric information**
- **stochasticity**
- **partner adaptation**
- **test-time protocol formation**

That makes it a much cleaner fit for benchmarking post-training methods that should improve coordination with unknown partners.

NanoCoop evaluates the focal policy primarily via **cross-play** against a fixed public partner zoo on held-out seeds and held-out layout slices.

---

## Status

This repo ships as a **full scaffold with runnable OvercookedV2 smoke baselines**:

- the **official benchmark target** is `jaxmarl` / OvercookedV2
- the first no-change policy baseline is `gpt-4.1-nano` in real LLM mode
- the Qwen3.5 configs remain available as the forward open-model target
- `records/` starts empty except for documentation; add rows only after verified OvercookedV2 runs

---

## Leaderboard

| Track | Rank | Run | Score | Summary | Record |
| --- | ---: | --- | ---: | --- | --- |
| `offline_20min_1xa100_40gb` | - | - | - | Awaiting verified OvercookedV2 run | - |
| `rlvr_20min_2xa100_40gb` | - | - | - | Awaiting verified OvercookedV2 run | - |
| `prompt_opt_1usd_gpt54_family` | - | - | - | Awaiting verified OvercookedV2 run | - |

New rows: add `records/<track>/<YYYY-MM-DD>_<name>/` and update this table in the same PR.

---

## Starter Agent Baseline

Before changing any training code, score the no-change focal agent:

```bash
./scripts/run_starter_agent_gpt41_nano.sh
```

Requires `OPENAI_API_KEY` in the environment.

This evaluates the seed coordination prompt in `gpt-4.1-nano` against the
public partner zoo on held-out OvercookedV2 layouts and seeds. The resulting
`cross_play_mean_reward` is the baseline-to-beat for method development.

The checked-in starter YAML remains a reference config for local tooling, but
`submission/agent.py` is now the official submission surface.

The starter agent is intentionally cheap and closed-model. NanoCoop should reward better coordination methods, data selection, rollout strategy, prompt search, and partner adaptation, not model-scale changes. The baseline asks the model for short cooperative macro-action plans, then executes those plans for several environment ticks before replanning. The run scripts stop after 180 seconds by default for fast iteration; set `NANOCOOP_TIMEOUT_SECONDS=0` for full benchmark records. The initial action set contains cooking and movement actions only; explicit communication actions are excluded until the environment adapter supports them as real state-changing actions.

The checked-in starter record under `records/starter_agent/` is the public no-change baseline reference. It is not a leaderboard submission for the training tracks.

---

## Official v0.1 contract

- Official v0.1 score uses the pinned default 20 cross-play episodes in config.
- The full 48-episode grid is available through the `episodes` command for diagnostics and future v0.2 expansion.
- Benchmark-eligible records must run with `NANOCOOP_TIMEOUT_SECONDS=0`.
- The default 180-second timeout is a developer guard only; timeout-marked runs are not official records.
- Config sweeps, model upgrades, layout edits, or seed edits are experimental unless the run is explicitly marked non-eligible.
- Qwen3.5 configs are forward-compatible targets and are not official v0.1 records until serving is verified.

---

## Change and run

Each track has **one Python file** containing the training algorithm and **one shell script** to run it.

### Offline (SFT) track

1. Change the training algorithm in `src/nanocoop/baselines/offline_sft.py`
2. Run:
   ```bash
   ./scripts/run_offline_training_gpt41_nano.sh
   ```

Budget: `20` minutes on `1x A100 40GB`
Student: `gpt-4.1-nano` starter policy path
Future open-model target: `Qwen/Qwen3.5-4B`

What the script handles:

- loads a partner zoo and train / held-out layout splits
- collects teacher traces
- filters traces into a compact post-training set
- builds a starter policy package
- runs held-out cross-play evaluation
- writes a record bundle

Smoke override:

```bash
NANOCOOP_OFFLINE_CONFIG=configs/offline_smoke.yaml \
./scripts/run_offline_training_gpt41_nano.sh
```

### RLVR track

1. Change the training algorithm in `src/nanocoop/baselines/rlvr.py`
2. Run:
   ```bash
   ./scripts/run_overcooked_rlvr_gpt41_nano_2xa100_20min.sh
   ```

Budget: `20` minutes on `2x A100 40GB`
Policy model: `gpt-4.1-nano`
Future open-model target: `Qwen/Qwen3.5-4B`

What the script handles:

- bootstraps a seed policy package
- runs grouped online rollouts against the partner zoo
- applies return-weighted updates
- writes periodic eval and final eval outputs
- materializes a record bundle

Smoke override:

```bash
NANOCOOP_RLVR_CONFIG=configs/rlvr_smoke.yaml \
./scripts/run_overcooked_rlvr_gpt41_nano_2xa100_20min.sh
```

### Prompt-opt track

1. Change the search / optimization algorithm in `src/nanocoop/baselines/prompt_opt.py`
2. Run:
   ```bash
   ./scripts/run_overcooked_prompt_opt_gpt41_nano_gpt54_budget.sh
   ```

Budget: `$1` optimizer spend (GPT-5.4 family)
Policy target: `gpt-4.1-nano`
Future open-model target: `Qwen/Qwen3.5-4B`

What the script handles:

- mutates the coordination prompt
- evaluates candidates on a small budget probe split
- keeps the best prompt / examples package
- runs final held-out eval and writes the bundle

Smoke override:

```bash
NANOCOOP_PROMPT_OPT_CONFIG=configs/prompt_opt_smoke.yaml \
./scripts/run_overcooked_prompt_opt_gpt41_nano_gpt54_budget.sh
```

---

## Benchmark score

Primary score:

```text
cross_play_mean_reward
```

This is the mean team reward when the focal policy is paired with the public evaluation partner zoo on held-out layout / seed combinations.

Checked-in summaries also include:

- `self_play_mean_reward`
- `mean_completion_rate`
- `cross_partner_std`
- `num_eval_episodes`
- per-layout and per-partner breakdowns
- failed episode diagnostics

The benchmark is intentionally **cross-play first**. High self-play and weak partner robustness is not the target.

---

## Policy package contract

Every training track outputs a `policy_package.json` with:

- `system_prompt`
- `behavior_flags`
- `fewshot_examples`
- `action_lookup`
- optional `adapter_path`
- metadata

That keeps the repo usable in two modes:

1. **Pure local OvercookedV2 smoke mode** with the included heuristic / lookup policies.
2. **Real LLM mode** with an OpenAI-compatible endpoint and optional external post-training artifacts.

This means you can start cheap and iterate fast, then swap in the real model serving stack without changing the benchmark contract.

---

## Repo layout

```text
nanocoop/
├── configs/
├── docs/
├── records/
├── scripts/
├── src/nanocoop/
│   ├── baselines/
│   └── envs/
└── tests/
```

Important docs:

- `docs/task-overcooked-v2.md`
- `docs/partner-zoo.md`
- `docs/tracks/offline_20min_1xa100_40gb.md`
- `docs/tracks/rlvr_20min_2xa100_40gb.md`
- `docs/tracks/prompt_opt_1usd_gpt54_family.md`
- `records/README.md`

---

## Quickstart

```bash
uv sync --extra overcookedv2 --group dev
uv run make smoke
```

---

## Compete

1. Pick a track.
2. Change the single baseline file for that track.
3. Run the track script.
4. Inspect `metrics.json` and `summary.md`.
5. Put the record in `records/<track>/<YYYY-MM-DD>_<name>/`.
6. Update the leaderboard table in this README in the same PR.

Good submissions improve **coordination**, not just one partner convention.
