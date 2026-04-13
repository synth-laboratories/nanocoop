from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nanocoop.eval_slice import format_table, run_eval, write_report
from nanocoop.policy import StarterPolicyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoCoop starter agent smoke run")
    parser.add_argument("--no-self-play", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", default="")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def load_config() -> StarterPolicyConfig:
    config_path = Path("configs/starter_agent_gpt41_nano_overcooked_v2.yaml")
    if not config_path.exists():
        return StarterPolicyConfig()
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return StarterPolicyConfig(
        model=values.get("model", "gpt-4.1-nano"),
        plan_horizon=int(values.get("plan_horizon", "4")),
        policy_decision_interval=int(values.get("policy_decision_interval", "8")),
        partner_aware_guardrails=values.get("partner_aware_guardrails", "true").lower()
        in {"1", "true", "yes", "on"},
        cooperation_prompt=values.get("cooperation_prompt", StarterPolicyConfig().cooperation_prompt),
    )


def main() -> int:
    _ = parse_args()
    config = load_config()

    baseline = run_eval(
        StarterPolicyConfig(
            model=config.model,
            plan_horizon=config.plan_horizon,
            policy_decision_interval=config.policy_decision_interval,
            partner_aware_guardrails=False,
            cooperation_prompt=config.cooperation_prompt,
        )
    )
    candidate = run_eval(config)

    out_path = Path("artifacts/honest_eval_slice.json")
    write_report(out_path, baseline, candidate)

    print("NanoCoop starter agent smoke eval")
    print(f"model={config.model}")
    print(f"plan_horizon={config.plan_horizon}")
    print(f"policy_decision_interval={config.policy_decision_interval}")
    print(f"partner_aware_guardrails={config.partner_aware_guardrails}")
    print()
    print("baseline,total_score=%s,max_score=%s" % (baseline["total_score"], baseline["max_score"]))
    print(format_table(baseline["rows"]))
    print()
    print("candidate,total_score=%s,max_score=%s" % (candidate["total_score"], candidate["max_score"]))
    print(format_table(candidate["rows"]))
    print()
    print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

