from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class StarterPolicyConfig:
    model: str = "gpt-4.1-nano"
    plan_horizon: int = 4
    policy_decision_interval: int = 8
    partner_aware_guardrails: bool = True
    cooperation_prompt: str = (
        "Act as a cooperative Overcooked partner: preserve the shared plan, "
        "avoid duplicating your partner's job, and prefer actions that unlock "
        "the next joint step when they are safe and locally useful."
    )
    no_op_actions: tuple[str, ...] = field(
        default_factory=lambda: ("noop", "no_op", "idle", "wait_if_unknown")
    )


def _available_actions(state: Mapping[str, Any]) -> list[str]:
    actions = state.get("available_actions")
    if isinstance(actions, list) and all(isinstance(item, str) for item in actions):
        return actions
    return [
        "advance_recipe",
        "support_partner",
        "clear_path",
        "prep_handoff",
        "hold_position",
    ]


def choose_action(state: Mapping[str, Any], config: StarterPolicyConfig) -> str:
    """Select a conservative, partner-aware starter action.

    The logic is intentionally narrow:
    - preserve the official starter settings
    - avoid duplicating partner work when a useful support action exists
    - avoid no-op communication actions
    - keep replans coarse enough to respect the 4-step horizon
    """

    actions = _available_actions(state)
    partner_busy = bool(state.get("partner_busy"))
    partner_needs_space = bool(state.get("partner_needs_space"))
    partner_has_needed_item = bool(state.get("partner_has_needed_item"))
    own_blocked = bool(state.get("own_blocked"))
    solo_progress_urgent = bool(state.get("solo_progress_urgent"))

    if config.partner_aware_guardrails:
        for candidate in ("clear_path", "prep_handoff", "support_partner"):
            if candidate in actions and (partner_needs_space or partner_has_needed_item):
                return candidate
        if partner_busy and "support_partner" in actions:
            return "support_partner"
        if own_blocked and "hold_position" in actions:
            return "hold_position"
        if solo_progress_urgent and "advance_recipe" in actions:
            return "advance_recipe"

    if "advance_recipe" in actions:
        return "advance_recipe"
    if "hold_position" in actions:
        return "hold_position"
    return actions[0]


def score_action(state: Mapping[str, Any], action: str) -> int:
    """Simple honest proxy score for the smoke eval slice."""

    target = state.get("preferred_action")
    if action == target:
        return 2

    support_targets = state.get("supportive_actions", [])
    if isinstance(support_targets, list) and action in support_targets:
        return 1

    if action in {"noop", "no_op", "idle"}:
        return -2

    return 0
