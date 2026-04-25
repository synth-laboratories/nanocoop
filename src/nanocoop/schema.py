from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


def _dungeongrid_invalid_correction_hint(reason: str, action: Any) -> str:
    action_type = action.get("type") if isinstance(action, dict) else None
    if reason == "blocked_movement":
        return "Do not repeat that direction from this tile; choose a different adjacent floor tile, open a nearby door, inspect, or end the turn."
    if reason == "illegal_target" and action_type == "inspect_tile":
        return "Inspect only nearby visible/reachable tiles; avoid far coordinates and tiles outside the visible map."
    if reason == "insufficient_ap":
        return "Use cheaper actions that fit remaining AP, or end_turn."
    if reason in {"illegal_action", "unknown_action_type"}:
        return "Use only documented action types and role-available spells/items."
    if reason == "missing_target":
        return "Include the required target field for this action type."
    if reason == "not_active_agent":
        return "Only submit actions for the active hero named in the observation."
    return "Change the plan instead of repeating this exact action."


@dataclass
class Observation:
    agent_id: str
    layout: str
    step_index: int
    max_steps: int
    private_recipe: str | None
    private_pot: str | None
    shared_recipe_known: bool
    shared_pot_known: bool
    ingredient_ready: bool
    pot_ready: bool
    dish_ready: bool
    plated: bool
    delivered: bool
    last_partner_action: str | None
    last_joint_event: str | None
    available_actions: tuple[str, ...]
    convention_hint: str | None = None
    recent_events: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        parts = [
            f"layout={self.layout}",
            f"step={self.step_index}",
            f"recipe={self.private_recipe or 'UNK'}",
            f"pot={self.private_pot or 'UNK'}",
            f"shared_recipe={int(self.shared_recipe_known)}",
            f"shared_pot={int(self.shared_pot_known)}",
            f"ingredient={int(self.ingredient_ready)}",
            f"pot_ready={int(self.pot_ready)}",
            f"dish={int(self.dish_ready)}",
            f"plated={int(self.plated)}",
            f"delivered={int(self.delivered)}",
            f"partner={self.last_partner_action or 'NONE'}",
            f"event={self.last_joint_event or 'NONE'}",
            f"hint={self.convention_hint or 'NONE'}",
        ]
        return "|".join(parts)

    def to_prompt(self) -> str:
        if self.metadata.get("backend") == "dungeongrid":
            return self._dungeongrid_prompt()
        recent = "\n".join(f"- {event}" for event in self.recent_events) or "- none"
        metadata = "\n".join(
            f"- {key}: {value}" for key, value in sorted(self.metadata.items())
        ) or "- none"
        return (
            f"layout: {self.layout}\n"
            f"step: {self.step_index}/{self.max_steps}\n"
            f"private_recipe: {self.private_recipe or 'unknown'}\n"
            f"private_pot: {self.private_pot or 'unknown'}\n"
            f"shared_recipe_known: {self.shared_recipe_known}\n"
            f"shared_pot_known: {self.shared_pot_known}\n"
            f"ingredient_ready: {self.ingredient_ready}\n"
            f"pot_ready: {self.pot_ready}\n"
            f"dish_ready: {self.dish_ready}\n"
            f"plated: {self.plated}\n"
            f"delivered: {self.delivered}\n"
            f"last_partner_action: {self.last_partner_action or 'none'}\n"
            f"last_joint_event: {self.last_joint_event or 'none'}\n"
            f"convention_hint: {self.convention_hint or 'none'}\n"
            f"kitchen_state:\n{metadata}\n"
            f"recent_events:\n{recent}\n"
            f"available_actions: {', '.join(self.available_actions)}"
        )

    def _dungeongrid_prompt(self) -> str:
        recent = "\n".join(f"- {event}" for event in self.recent_events) or "- none"
        visible_map = self.metadata.get("visible_map_coordinates") or self.metadata.get("visible_map") or []
        if isinstance(visible_map, list):
            visible_map_text = "\n".join(str(row) for row in visible_map) or "(none)"
        else:
            visible_map_text = str(visible_map)
        self_state = self.metadata.get("self") or {}
        objective = self.metadata.get("objective") or self.metadata.get("quest_objective") or "unknown"
        visible_entities = self.metadata.get("visible_entities") or self.metadata.get("entities") or []
        visible_objects = self.metadata.get("visible_objects") or []
        visible_rooms = self.metadata.get("visible_rooms") or []
        party_roster = self.metadata.get("party_roster") or []
        visible_teammates = self.metadata.get("visible_teammates") or []
        adjacent_tiles = self.metadata.get("adjacent_tiles") or []
        party_messages = self.metadata.get("party_messages") or []
        invalid_feedback = self.metadata.get("invalid_feedback") or []
        inventory = self.metadata.get("inventory") or (
            self_state.get("inventory") if isinstance(self_state, dict) else None
        )
        hp = self_state.get("hp") if isinstance(self_state, dict) else self.metadata.get("hp")
        pos = self_state.get("pos") if isinstance(self_state, dict) else None
        equipment = self_state.get("equipment") if isinstance(self_state, dict) else None
        spell_cards = []
        used_spell_cards = []
        if isinstance(equipment, dict):
            used_spell_cards = [str(card) for card in equipment.get("used_spell_cards", [])]
            used_counts: dict[str, int] = {}
            for card in used_spell_cards:
                used_counts[card] = used_counts.get(card, 0) + 1
            for card in equipment.get("spell_cards", []):
                key = str(card)
                if used_counts.get(key, 0) > 0:
                    used_counts[key] -= 1
                    continue
                spell_cards.append(key)
        role = self.metadata.get("role") or (
            self_state.get("role") if isinstance(self_state, dict) else None
        )
        invalid_warning = self._dungeongrid_invalid_feedback_text(invalid_feedback)
        return (
            f"track: dungeongrid\n"
            f"quest: {self.layout}\n"
            f"turn: {self.step_index}/{self.max_steps}\n"
            f"active_agent: {self.metadata.get('active_agent', self.agent_id)}\n"
            f"active_role: {role or 'unknown'}\n"
            f"hp: {hp if hp is not None else 'unknown'}\n"
            f"ap_remaining: {self.metadata.get('ap_remaining', 'unknown')}\n"
            "action_costs: move/open_door/interact/use_item/equip_item/give_item/message/guard=1 AP; "
            "attack_melee/attack_ranged/cast/disarm/inspect_room=2 AP; "
            "end_turn=0 AP. Keep the submitted plan within current AP unless a "
            "reveal boundary is likely to stop it early.\n"
            "plan_safety: later actions are checked from the hero's new position "
            "after earlier moves. Queue chained moves only when the visible map makes "
            "the route clear; otherwise move into a better position and replan on the "
            "next observation. Do not queue an interaction with an object whose "
            "affordance says visible_not_adjacent or visible_closed_door_not_adjacent.\n"
            f"active_position: {pos if pos is not None else 'unknown'}\n"
            f"equipment: {equipment or 'unknown'}\n"
            f"spell_cards_available: {spell_cards or 'none'}\n"
            f"spell_cards_used: {used_spell_cards or 'none'}\n"
            f"objective: {objective}\n"
            f"inventory: {inventory or 'empty'}\n"
            f"party_roster:\n{json.dumps(party_roster, indent=2, sort_keys=True)}\n"
            f"visible_teammates:\n{json.dumps(visible_teammates, indent=2, sort_keys=True)}\n"
            f"adjacent_tiles:\n{json.dumps(adjacent_tiles, indent=2, sort_keys=True)}\n"
            f"previous_invalid_actions_to_correct:\n{invalid_warning}\n"
            f"last_warden_action: {self.last_partner_action or 'none'}\n"
            f"last_hero_action: {self.last_joint_event or 'none'}\n"
            f"visible_map:\n{visible_map_text}\n"
            f"visible_entities:\n{json.dumps(visible_entities, sort_keys=True)}\n"
            f"visible_objects:\n{json.dumps(visible_objects, indent=2, sort_keys=True)}\n"
            f"visible_rooms:\n{json.dumps(visible_rooms, indent=2, sort_keys=True)}\n"
            f"party_messages:\n{json.dumps(party_messages, indent=2, sort_keys=True)}\n"
            f"invalid_feedback:\n{json.dumps(invalid_feedback, indent=2, sort_keys=True)}\n"
            f"recent_events:\n{recent}\n"
            "Use dungeongrid_act to submit JSON action objects. "
            "Use the rules/tool schema for mechanics; legal actions are validated by the environment."
        )

    def _dungeongrid_invalid_feedback_text(self, invalid_feedback: Any) -> str:
        if not invalid_feedback:
            return "- none"
        if not isinstance(invalid_feedback, list):
            return json.dumps(invalid_feedback, indent=2, sort_keys=True)
        lines: list[str] = []
        for feedback in invalid_feedback[-5:]:
            if not isinstance(feedback, dict):
                continue
            action = feedback.get("action") or {}
            reason = str(feedback.get("reason") or "invalid")
            message = str(feedback.get("message") or "")
            hint = _dungeongrid_invalid_correction_hint(reason, action)
            lines.append(
                f"- {feedback.get('agent_id', 'unknown')}: action="
                f"{json.dumps(action, sort_keys=True)} failed with {reason}: {message} "
                f"Correction: {hint}"
            )
        return "\n".join(lines) or "- none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    step_index: int
    action_agent_0: str
    action_agent_1: str
    reward: float
    event: str
    focal_observation: Observation

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["focal_observation"] = self.focal_observation.to_dict()
        return data


@dataclass
class EpisodeTrace:
    layout: str
    seed: int
    partner_name: str
    total_reward: float
    success: bool
    steps: list[StepRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "seed": self.seed,
            "partner_name": self.partner_name,
            "total_reward": self.total_reward,
            "success": self.success,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }


@dataclass
class PolicyPackage:
    name: str
    backend: str
    system_prompt: str
    behavior_flags: list[str]
    fewshot_examples: list[dict[str, Any]] = field(default_factory=list)
    action_lookup: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    adapter_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyPackage":
        return cls(**data)


@dataclass
class EvalEpisodeResult:
    layout: str
    partner_name: str
    seed: int
    total_reward: float
    success: bool
    mode: str
    episode_id: int | None = None
    step_count: int | None = None
    llm_call_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
