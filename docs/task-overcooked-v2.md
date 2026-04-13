# Task Overcooked V2

This repo tracks the starter-policy candidate for the NanoCoop leaderboard.

Constraints:
- keep `gpt-4.1-nano`
- keep `plan_horizon: 4`
- keep `policy_decision_interval: 8`
- improve cooperation with a narrow prompt or action-selection guardrail
- do not reintroduce mock backends or no-op communication actions

The current candidate adds a small partner-aware guardrail:
- prefer `clear_path`, `prep_handoff`, or `support_partner` when the partner is
  already doing the critical ingredient movement
- fall back to `advance_recipe` only when the state does not call for support
- avoid no-op actions in the starter policy surface

Validation:
- `scripts/run_starter_agent_gpt41_nano.sh` writes a deterministic smoke eval
  to `artifacts/honest_eval_slice.json`
- the eval compares the candidate against a no-guardrail baseline on a small
  cooperative scenario slice

