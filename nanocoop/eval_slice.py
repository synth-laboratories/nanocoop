from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from nanocoop.policy import StarterPolicyConfig, choose_action, score_action


SCENARIOS = [
    {
        "name": "partner-needs-space",
        "partner_busy": True,
        "partner_needs_space": True,
        "available_actions": ["advance_recipe", "support_partner", "clear_path"],
        "preferred_action": "clear_path",
        "supportive_actions": ["support_partner"],
    },
    {
        "name": "partner-has-needed-item",
        "partner_has_needed_item": True,
        "available_actions": ["advance_recipe", "prep_handoff", "support_partner"],
        "preferred_action": "prep_handoff",
        "supportive_actions": ["support_partner"],
    },
    {
        "name": "own-blocked",
        "own_blocked": True,
        "available_actions": ["advance_recipe", "hold_position", "support_partner"],
        "preferred_action": "hold_position",
        "supportive_actions": ["support_partner"],
    },
    {
        "name": "solo-progress-urgent",
        "solo_progress_urgent": True,
        "available_actions": ["advance_recipe", "support_partner", "clear_path"],
        "preferred_action": "advance_recipe",
        "supportive_actions": ["support_partner", "clear_path"],
    },
    {
        "name": "neutral-coordination",
        "available_actions": ["advance_recipe", "support_partner", "clear_path"],
        "preferred_action": "advance_recipe",
        "supportive_actions": ["support_partner"],
    },
]


def run_eval(config: StarterPolicyConfig) -> dict[str, object]:
    rows = []
    total = 0
    for scenario in SCENARIOS:
        action = choose_action(scenario, config)
        score = score_action(scenario, action)
        total += score
        rows.append(
            {
                "name": scenario["name"],
                "action": action,
                "preferred": scenario["preferred_action"],
                "score": score,
            }
        )
    return {
        "config": asdict(config),
        "total_score": total,
        "max_score": 10,
        "rows": rows,
    }


def write_report(path: Path, baseline: dict[str, object], candidate: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"baseline": baseline, "candidate": candidate}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_table(rows: Iterable[dict[str, object]]) -> str:
    lines = ["scenario,action,preferred,score"]
    for row in rows:
        lines.append(
            f"{row['name']},{row['action']},{row['preferred']},{row['score']}"
        )
    return "\n".join(lines)

