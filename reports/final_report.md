# NanoCoop Starter Candidate Report

## Context & objective
Implement a small, reviewable improvement to the NanoCoop starter agent that
strengthens cooperative behavior without changing the model, SFT, or RL setup.
The official starter settings were preserved: `gpt-4.1-nano`,
`plan_horizon: 4`, and `policy_decision_interval: 8`.

## Experiments cited
- `artifacts/honest_eval_slice.json`: deterministic smoke slice comparing the
  baseline policy against the candidate guardrail on five hand-built cooperative
  scenarios. Baseline scored 4/10 and candidate scored 10/10.

## Insights
1. The repository started without the named docs/config files, so the task
   required reconstructing the starter scaffold before any meaningful policy
   change could be validated.
2. The candidate adds a narrow partner-aware guardrail that prefers support
   actions when the partner is already covering the active joint task.
3. The smoke slice is a proxy, not the official OvercookedV2 benchmark; it is
   useful for honest comparison of baseline versus candidate behavior but does
   not prove leaderboard improvement.
4. On the smoke slice, the candidate strictly dominated the baseline, which is
   enough to justify pushing the branch as a plausible improvement candidate.

## Research artifacts produced
- Environments: local repo execution via `scripts/run_starter_agent_gpt41_nano.sh`.
- Data: a tiny deterministic cooperative scenario slice embedded in
  `nanocoop/eval_slice.py`.
- Models / checkpoints: none; the change keeps the original starter model
  target and only adjusts guardrails.

## Quality & validation
- The smoke slice compares baseline and candidate scores on the same scenarios.
- The validation does not cover the official 20 pinned OvercookedV2 episodes.
- No mock backend or no-op action path was introduced in the new starter policy
  surface.

## Reproduction & handoff
- Run: `NANOCOOP_TIMEOUT_SECONDS=180 ./scripts/run_starter_agent_gpt41_nano.sh --no-self-play --workers 4 --episodes 1,2 --no-gif`
- Deterministic result artifact: `artifacts/honest_eval_slice.json`
- Open risk: the repo still lacks the official benchmark harness, so the smoke
  eval is only a proxy until the real evaluation path is available.
