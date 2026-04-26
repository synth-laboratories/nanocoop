from __future__ import annotations

import difflib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol

import requests
from dungeongrid import (
    DungeonGridAction,
    WardenDecision,
    WardenReActAdapter,
    dungeongrid_act_schema,
    dungeongrid_rules,
    dungeongrid_rules_schema,
    dungeongrid_warden_act_schema,
)
from pydantic import ValidationError

from nanocoop.prompts import extract_behavior_flags, render_fewshot_examples
from nanocoop.schema import Observation, PolicyPackage


LLM_USAGE_TOKEN_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)


def _new_llm_usage_totals() -> dict[str, Any]:
    return {
        "requests_with_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }


def _record_llm_usage(
    totals: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    model: str,
    body: dict[str, Any],
) -> None:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return
    totals["requests_with_usage"] = int(totals.get("requests_with_usage", 0) or 0) + 1
    for key in LLM_USAGE_TOKEN_KEYS:
        value = usage.get(key)
        if isinstance(value, int | float):
            totals[key] = int(totals.get(key, 0) or 0) + int(value)

    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(prompt_details, dict):
        cached = prompt_details.get("cached_tokens")
        if isinstance(cached, int | float):
            totals["cached_tokens"] = int(totals.get("cached_tokens", 0) or 0) + int(cached)

    completion_details = (
        usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    )
    if isinstance(completion_details, dict):
        reasoning = completion_details.get("reasoning_tokens")
        if isinstance(reasoning, int | float):
            totals["reasoning_tokens"] = int(totals.get("reasoning_tokens", 0) or 0) + int(reasoning)

    choice = (body.get("choices") or [{}])[0]
    events.append(
        {
            "finish_reason": choice.get("finish_reason"),
            "model": model,
            "usage": usage,
        }
    )


class Policy(Protocol):
    def act(self, observation: Observation) -> str:
        ...

    def act_plan(self, observation: Observation) -> list[Any]:
        ...


@dataclass
class HybridLookupPolicy:
    package: PolicyPackage
    exploration_rate: float = 0.0
    rng_seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)
        self._flags = set(self.package.behavior_flags)

    def act(self, observation: Observation) -> str:
        if self.exploration_rate > 0 and self._rng.random() < self.exploration_rate:
            return self._rng.choice(list(observation.available_actions))
        signature = observation.signature()
        if signature in self.package.action_lookup:
            return self.package.action_lookup[signature]
        return self._heuristic_action(observation)

    def act_plan(self, observation: Observation) -> list[Any]:
        if observation.metadata.get("backend") == "dungeongrid":
            legal = list(observation.metadata.get("internal_action_objects") or [])
            if legal:
                return [self._dungeongrid_heuristic_action(observation, legal)]
            return self._dungeongrid_heuristic_plan(observation)
        return [self.act(observation)]

    def _dungeongrid_heuristic_plan(self, observation: Observation) -> list[dict[str, Any]]:
        self_state = observation.metadata.get("self") or {}
        pos = self_state.get("pos") if isinstance(self_state, dict) else None
        visible_objects = observation.metadata.get("visible_objects") or []
        visible_entities = observation.metadata.get("visible_entities") or []
        role = str(observation.metadata.get("role") or "")
        plan: list[dict[str, Any]] = []

        for ent in visible_entities:
            if ent.get("team") == "dungeon" and ent.get("id"):
                if role == "wizard":
                    return [{"type": "cast", "target": ent["id"], "payload": {"spell": "spark_lance"}}]
                return [{"type": "attack_melee", "target": ent["id"]}, {"type": "attack_ranged", "target": ent["id"]}]

        for obj in visible_objects:
            if obj.get("type") == "objective" and obj.get("id"):
                plan.extend(self._move_toward_adjacent(pos, obj.get("pos")))
                plan.append({"type": "interact", "target": obj["id"]})
                return plan[:4]
            if obj.get("type") == "chest" and obj.get("id"):
                plan.extend(self._move_toward_adjacent(pos, obj.get("pos")))
                plan.append({"type": "interact", "target": obj["id"]})
                return plan[:4]
            if obj.get("type") == "door" and obj.get("state") == "closed" and obj.get("id"):
                plan.extend(self._move_toward_adjacent(pos, obj.get("pos")))
                plan.append({"type": "open_door", "target": obj["id"]})
                return plan[:4]
            if obj.get("type") == "trap" and role == "dwarf" and obj.get("id"):
                plan.extend(self._move_toward_adjacent(pos, obj.get("pos")))
                plan.append({"type": "disarm", "target": obj["id"]})
                return plan[:3]

        return [
            {"type": "inspect_room"},
            {"type": "move", "direction": "east"},
            {"type": "move", "direction": "south"},
            {"type": "end_turn"},
        ]

    def _move_toward_adjacent(self, pos: Any, target: Any) -> list[dict[str, Any]]:
        if not (
            isinstance(pos, list)
            and len(pos) == 2
            and isinstance(target, list)
            and len(target) == 2
        ):
            return []
        x, y = int(pos[0]), int(pos[1])
        tx, ty = int(target[0]), int(target[1])
        plan: list[dict[str, Any]] = []
        while abs(x - tx) + abs(y - ty) > 1 and len(plan) < 3:
            if abs(tx - x) >= abs(ty - y) and tx != x:
                direction = "east" if tx > x else "west"
                x += 1 if tx > x else -1
            elif ty != y:
                direction = "south" if ty > y else "north"
                y += 1 if ty > y else -1
            else:
                break
            plan.append({"type": "move", "direction": direction})
        return plan

    def _dungeongrid_heuristic_action(
        self, observation: Observation, legal: list[dict[str, Any]]
    ) -> dict[str, Any]:
        role = str(observation.metadata.get("role") or "")
        priorities = (
            ("interact", "escape"),
            ("interact",),
            ("cast",) if role in {"wizard", "elf"} else ("attack_melee",),
            ("attack_melee",),
            ("attack_ranged",),
            ("disarm",) if role == "dwarf" else ("open_door",),
            ("open_door",),
            ("message",),
            ("inspect_room",),
            ("move",),
            ("inspect_tile",),
            ("guard",),
            ("end_turn",),
        )
        for priority in priorities:
            for action in legal:
                if action.get("type") != priority[0]:
                    continue
                if len(priority) == 1 or action.get("target") == priority[1]:
                    return action
        return legal[0]

    def _heuristic_action(self, observation: Observation) -> str:
        carrying = str(observation.metadata.get("inventory", "empty"))
        if "cooked_soup" in carrying and "plate" in carrying:
            return "SERVE_SOUP"
        if "ingredient" in carrying:
            return "PREP_POT"
        if "plate" in carrying and observation.pot_ready:
            return "PLATE_SOUP"
        can_reach_ingredient = bool(observation.metadata.get("can_reach_ingredient", True))
        pot_count = int(observation.metadata.get("pot_ingredient_count", 0))
        pot_full = bool(observation.metadata.get("pot_full", pot_count >= 3))

        if (
            "infer_partner_convention" in self._flags
            and observation.last_partner_action == "FETCH_DISH"
            and not pot_full
            and can_reach_ingredient
        ):
            return "FETCH_INGREDIENT"
        if (
            "infer_partner_convention" in self._flags
            and observation.last_partner_action == "PREP_POT"
            and not pot_full
            and can_reach_ingredient
        ):
            return "FETCH_INGREDIENT"

        if (
            observation.ingredient_ready
            and observation.pot_ready
            and observation.dish_ready
            and not observation.plated
        ):
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"

        if (
            "avoid_duplicate_work" in self._flags
            and observation.last_partner_action == "FETCH_DISH"
        ):
            if not pot_full and can_reach_ingredient:
                return "FETCH_INGREDIENT"
            if not observation.pot_ready:
                return "PREP_POT"

        if not pot_full and can_reach_ingredient:
            return "FETCH_INGREDIENT"
        if not observation.pot_ready:
            return "PREP_POT"
        if not observation.dish_ready:
            return "FETCH_DISH"
        return "WAIT"


@dataclass
class OracleTeacherPolicy:
    agent_id: str = "agent_0"

    def act(self, observation: Observation) -> str:
        carrying = str(observation.metadata.get("inventory", "empty"))
        if "cooked_soup" in carrying and "plate" in carrying:
            return "SERVE_SOUP"
        if "ingredient" in carrying:
            return "PREP_POT"
        if "plate" in carrying and observation.pot_ready:
            return "PLATE_SOUP"
        can_reach_ingredient = bool(observation.metadata.get("can_reach_ingredient", True))
        pot_count = int(observation.metadata.get("pot_ingredient_count", 0))
        pot_full = bool(observation.metadata.get("pot_full", pot_count >= 3))
        if (
            observation.ingredient_ready
            and observation.pot_ready
            and observation.dish_ready
            and not observation.plated
        ):
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"

        if (
            observation.last_partner_action == "FETCH_DISH"
            and not pot_full
            and can_reach_ingredient
        ):
            return "FETCH_INGREDIENT"
        if (
            observation.last_partner_action == "PREP_POT"
            and not pot_full
            and can_reach_ingredient
        ):
            return "FETCH_INGREDIENT"

        if not pot_full and can_reach_ingredient:
            return "FETCH_INGREDIENT"
        if not observation.pot_ready:
            return "PREP_POT"
        if not observation.dish_ready:
            return "FETCH_DISH"
        return "WAIT"

    def act_plan(self, observation: Observation) -> list[Any]:
        if observation.metadata.get("backend") == "dungeongrid":
            legal = list(observation.metadata.get("internal_action_objects") or [])
            if not legal:
                return HybridLookupPolicy(
                    PolicyPackage(name="oracle_dungeongrid", backend="dungeongrid", system_prompt="", behavior_flags=[])
                )._dungeongrid_heuristic_plan(observation)
            for action_type in (
                "interact",
                "attack_melee",
                "cast",
                "attack_ranged",
                "disarm",
                "open_door",
                "inspect_room",
                "move",
                "guard",
                "end_turn",
            ):
                for action in legal:
                    if action.get("type") == action_type:
                        return [action]
            return legal[:1] or [{"type": "end_turn"}]
        return [self.act(observation)]


@dataclass
class RemoteChatPolicy:
    package: PolicyPackage
    model_name: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_seconds: float = 30.0
    plan_horizon: int = 1

    def act(self, observation: Observation) -> str:
        if observation.metadata.get("backend") == "dungeongrid":
            plan = self.act_plan(observation)
            return json.dumps(plan[0], sort_keys=True) if plan else '{"type":"end_turn"}'
        if self.plan_horizon > 1:
            return self._act_from_plan(observation)
        return self._act_once(observation)

    def act_plan(self, observation: Observation) -> list[Any]:
        if observation.metadata.get("backend") == "dungeongrid":
            return self._request_dungeongrid_action_plan(observation)
        if self.plan_horizon > 1:
            return self._request_action_plan(observation)
        return [self._act_once(observation)]

    def __post_init__(self) -> None:
        self._planned_actions: list[str] = []
        self._last_plan_step = -1
        self._plan_state_key: tuple[Any, ...] | None = None
        self.llm_call_count = 0
        self.llm_usage = _new_llm_usage_totals()
        self.llm_usage_events: list[dict[str, Any]] = []

    def _act_from_plan(self, observation: Observation) -> str:
        if observation.step_index <= self._last_plan_step:
            self._planned_actions = []
            self._plan_state_key = None
        self._last_plan_step = observation.step_index
        override = self._override_action(observation)
        if override is not None:
            return override
        state_key = self._state_key(observation)
        if self._plan_state_key is not None and state_key != self._plan_state_key:
            self._planned_actions = []
        if not self._planned_actions:
            self._planned_actions = self._request_action_plan(observation)
            self._plan_state_key = state_key
        if not self._planned_actions:
            return "WAIT"
        action = self._planned_actions.pop(0)
        override = self._override_action(observation, planned_action=action)
        if override is not None:
            return override
        return action

    def _act_once(self, observation: Observation) -> str:
        prompt = observation.to_prompt()
        examples_text = render_fewshot_examples(self.package.fewshot_examples[:4])
        system_prompt = self.package.system_prompt.strip()
        if examples_text:
            system_prompt = f"{system_prompt}\n\nFew-shot examples:\n{examples_text}"

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        f"Respond with JSON: {{\"action\": \"...\"}} where action is one of "
                        f"{list(observation.available_actions)}."
                    ),
                },
            ],
        }
        response = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        self.llm_call_count += 1
        response.raise_for_status()
        body = response.json()
        _record_llm_usage(
            self.llm_usage,
            self.llm_usage_events,
            model=self.model_name,
            body=body,
        )
        content = body["choices"][0]["message"]["content"]
        return _extract_action_from_text(content)

    def _request_action_plan(self, observation: Observation) -> list[str]:
        prompt = observation.to_prompt()
        examples_text = render_fewshot_examples(self.package.fewshot_examples[:4])
        system_prompt = self.package.system_prompt.strip()
        if examples_text:
            system_prompt = f"{system_prompt}\n\nFew-shot examples:\n{examples_text}"

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        "Choose a short cooperative macro policy for the next "
                        f"{self.plan_horizon} decision points. The environment will "
                        "re-check the kitchen between actions, but you should commit "
                        "to a stable plan that complements the partner instead of "
                        "thrashing.\n\n"
                        "Respond with JSON: "
                        '{"actions": ["...", "..."]} where each action is one of '
                        f"{list(observation.available_actions)}. Return exactly "
                        f"{self.plan_horizon} actions."
                    ),
                },
            ],
        }
        response = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        self.llm_call_count += 1
        response.raise_for_status()
        body = response.json()
        _record_llm_usage(
            self.llm_usage,
            self.llm_usage_events,
            model=self.model_name,
            body=body,
        )
        content = body["choices"][0]["message"]["content"]
        return _extract_actions_from_text(content, limit=self.plan_horizon)

    def _request_dungeongrid_action_plan(self, observation: Observation) -> list[dict[str, Any]]:
        legal = list(observation.metadata.get("internal_action_objects") or [])
        if not legal:
            return [{"type": "end_turn"}]
        prompt = observation.to_prompt()
        examples_text = render_fewshot_examples(self.package.fewshot_examples[:4])
        system_prompt = self.package.system_prompt.strip()
        if examples_text:
            system_prompt = f"{system_prompt}\n\nFew-shot examples:\n{examples_text}"

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        "Submit this hero turn as a JSON list of action objects. "
                        "Use documented DungeonGrid action types and target ids from the visible state. "
                        "The environment will execute the list until AP or the turn ends, "
                        "skip actions that become illegal, and ignore unused extras.\n\n"
                        'Respond with JSON: {"actions": [{...}, {...}]}'
                    ),
                },
            ],
        }
        response = requests.post(
            f"{self.api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        self.llm_call_count += 1
        response.raise_for_status()
        body = response.json()
        _record_llm_usage(
            self.llm_usage,
            self.llm_usage_events,
            model=self.model_name,
            body=body,
        )
        content = body["choices"][0]["message"]["content"]
        actions = _extract_action_objects_from_text(content)
        return actions or [legal[-1]]

    def _state_key(self, observation: Observation) -> tuple[Any, ...]:
        return (
            observation.plated,
            observation.delivered,
            observation.pot_ready,
            observation.dish_ready,
            observation.ingredient_ready,
            str(observation.metadata.get("inventory", "empty")),
            observation.last_partner_action,
            observation.last_joint_event,
        )

    def _override_action(
        self, observation: Observation, planned_action: str | None = None
    ) -> str | None:
        carrying = str(observation.metadata.get("inventory", "empty"))
        can_reach_ingredient = bool(observation.metadata.get("can_reach_ingredient", True))
        can_reach_plate = bool(observation.metadata.get("can_reach_plate", True))
        partner_name = str(observation.metadata.get("partner_name", ""))
        pot_full = bool(observation.metadata.get("pot_full", False))

        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
        if "ingredient" in carrying:
            return "PREP_POT"
        if "plate" in carrying and observation.pot_ready:
            return "PLATE_SOUP"
        if "cooked_soup" in carrying and "plate" in carrying:
            return "SERVE_SOUP"

        partner_action = observation.last_partner_action
        if partner_action in {"FETCH_INGREDIENT", "PREP_POT"}:
            if not observation.dish_ready and can_reach_plate:
                return "FETCH_DISH"
            if observation.pot_ready and observation.dish_ready and not observation.plated:
                return "PLATE_SOUP"
        if partner_action in {"FETCH_DISH", "PLATE_SOUP", "SERVE_SOUP"}:
            if not pot_full and can_reach_ingredient:
                return "FETCH_INGREDIENT"
            if not observation.pot_ready:
                return "PREP_POT"

        if partner_name == "potter" and not observation.dish_ready and can_reach_plate:
            return "FETCH_DISH"
        if partner_name == "courier" and not pot_full and can_reach_ingredient:
            return "FETCH_INGREDIENT"
        if (
            partner_name == "handoff"
            and planned_action is not None
            and planned_action == partner_action
        ):
            if planned_action == "FETCH_INGREDIENT" and not observation.dish_ready and can_reach_plate:
                return "FETCH_DISH"
            if planned_action == "FETCH_DISH" and not pot_full and can_reach_ingredient:
                return "FETCH_INGREDIENT"
            if planned_action == "PREP_POT" and not observation.dish_ready and can_reach_plate:
                return "FETCH_DISH"

        if planned_action == "FETCH_INGREDIENT" and not can_reach_ingredient:
            if not observation.pot_ready:
                return "PREP_POT"
            if not observation.dish_ready:
                return "FETCH_DISH"
            return "WAIT"
        return None

    @classmethod
    def from_config(
        cls, package: PolicyPackage, config: dict[str, Any]
    ) -> "RemoteChatPolicy":
        model_cfg = config.get("model", {})
        default_api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        api_base = str(model_cfg.get("api_base") or default_api_base)
        api_key = os.getenv("OPENAI_API_KEY", model_cfg.get("api_key", "changeme"))
        return cls(
            package=package,
            model_name=str(model_cfg.get("name", package.model or "unknown-model")),
            api_base=api_base,
            api_key=api_key,
            temperature=float(model_cfg.get("temperature", 0.0)),
            max_tokens=int(model_cfg.get("max_tokens", 128)),
            plan_horizon=max(1, int(model_cfg.get("plan_horizon", 1))),
        )


@dataclass
class DungeonGridReActPolicy:
    package: PolicyPackage
    model_name: str
    api_base: str
    api_key: str
    temperature: float | None = None
    max_tokens: int = 768
    timeout_seconds: float = 60.0
    token_limit_field: str | None = None
    reasoning_effort: str | None = None
    omit_temperature: bool = False
    max_retries: int = 2
    max_tool_rounds: int = 6

    def __post_init__(self) -> None:
        self.llm_call_count = 0
        self.llm_usage = _new_llm_usage_totals()
        self.llm_usage_events: list[dict[str, Any]] = []
        self._private_plans: dict[str, str] = {}
        self.private_plan_tool_count = 0
        self.private_plan_tool_counts: dict[str, int] = {}

    def act(self, observation: Observation) -> str:
        plan = self.act_plan(observation)
        return json.dumps(plan[0], sort_keys=True) if plan else '{"type":"end_turn"}'

    def act_plan(self, observation: Observation) -> list[dict[str, Any]]:
        if observation.metadata.get("backend") != "dungeongrid":
            return [{"type": self._fallback_action_name(observation)}]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(observation)},
            {
                "role": "user",
                "content": (
                    f"{observation.to_prompt()}\n\n"
                    "Think privately. You may call dungeongrid_rules as many times as "
                    "needed to check mechanics. You may also use private plan tools to "
                    "read, write, append, or edit a medium-term plan that only this hero "
                    "can see across turns. Keep that plan short and operational: current "
                    "goal, route, next door/object, teammate coordination, and what to "
                    "avoid repeating. Revise your plan internally, then finish "
                    "with exactly one dungeongrid_act tool call. Submit a long OpenEnv "
                    "ReAct-style JSON action plan for only the active hero. The environment "
                    "will validate each proposed action, skip illegal queued actions, and "
                    "report concrete feedback in the next observation. Before planning, "
                    "read previous_invalid_actions_to_correct. Do not repeat failed moves "
                    "from the same tile, far inspect_tile targets, unknown spells, or "
                    "actions that exceed AP. When using a message action in multi-hero "
                    "runs, target party or a hero id from party_roster and add payload.text "
                    "with the exact instruction or report you want them to see. Use coordinate "
                    "targets only for inspect_tile. For open_door, interact, attack, cast, "
                    "and disarm, target the exact object/entity id shown in visible_objects "
                    "or visible_entities, such as door_1 or skitterling_1. Do not "
                    "start with end_turn when move, interact, inspect, attack, cast, "
                    "open_door, or disarm actions can make progress. Include end_turn "
                    "only when ending deliberately."
                ),
            },
        ]
        body: dict[str, Any] = {}
        for round_index in range(max(1, self.max_tool_rounds)):
            force_act = round_index == self.max_tool_rounds - 1
            payload = self._chat_payload(
                messages,
                tool_choice={"type": "function", "function": {"name": "dungeongrid_act"}}
                if force_act
                else "auto",
            )
            body = self._post_chat_completions(payload)
            if self._completion_budget_exhausted_without_output(body):
                raise RuntimeError(
                    "DungeonGrid ReAct model response exhausted max_completion_tokens "
                    "without visible content or tool calls. For reasoning-capable models, set "
                    "model.reasoning_effort: low/minimal or increase model.max_tokens."
                )
            message = (body.get("choices") or [{}])[0].get("message") or {}
            actions = self._extract_tool_actions(body)
            if actions:
                return actions
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                text_actions = _extract_action_objects_from_text(str(message.get("content") or ""))
                if text_actions:
                    return text_actions
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You did not call a tool. You may call dungeongrid_rules for "
                            "more information, but you must finish with dungeongrid_act."
                        ),
                    }
                )
                continue
            messages.append(self._assistant_tool_call_message(message))
            handled_any = False
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name")
                if name == "dungeongrid_rules":
                    handled_any = True
                    messages.append(self._rules_tool_result_message(tool_call))
                elif name in self._private_plan_tool_names():
                    handled_any = True
                    messages.append(self._private_plan_tool_result_message(tool_call, observation))
                elif name == "dungeongrid_act":
                    # Invalid/empty act arguments were already inspected above. Tell
                    # the model to repair the final call if there is room left.
                    handled_any = True
                    messages.append(
                        self._tool_result_message(
                            tool_call,
                            {"error": "dungeongrid_act actions must be a non-empty list of action objects."},
                        )
                    )
                else:
                    handled_any = True
                    messages.append(self._tool_result_message(tool_call, {"error": f"unknown tool: {name}"}))
            if not handled_any:
                break
        return _extract_action_objects_from_text(
            str((body.get("choices") or [{}])[0].get("message", {}).get("content") or "")
        ) or [{"type": "end_turn"}]

    def _chat_payload(self, messages: list[dict[str, Any]], *, tool_choice: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            **self._token_limit_payload(),
            "messages": messages,
            "tools": [
                self._tool_schema(),
                dungeongrid_rules_schema(),
                *self._private_plan_tool_schemas(),
            ],
            "tool_choice": tool_choice,
        }
        if not self.omit_temperature and self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: requests.RequestException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.api_base.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                self.llm_call_count += 1
                response.raise_for_status()
                body = response.json()
                _record_llm_usage(
                    self.llm_usage,
                    self.llm_usage_events,
                    model=self.model_name,
                    body=body,
                )
                return body
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
        raise RuntimeError("unreachable DungeonGrid ReAct retry state") from last_error

    def _completion_budget_exhausted_without_output(self, body: dict[str, Any]) -> bool:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0
        return (
            choice.get("finish_reason") == "length"
            and not content
            and not tool_calls
            and reasoning_tokens > 0
        )

    def _system_prompt(self, observation: Observation | None = None) -> str:
        base = self.package.system_prompt.strip()
        rules = "\n".join(
            f"- {topic}: {dungeongrid_rules(topic)}"
            for topic in (
                "turn",
                "actions",
                "action_contract",
                "movement",
                "combat",
                "spells",
                "cards",
                "items",
                "objects",
                "traps_locks",
                "doors_rooms",
                "messages",
                "scoring",
            )
        )
        roster_text = ""
        if observation is not None:
            roster = observation.metadata.get("party_roster") or []
            if roster:
                roster_lines = []
                for member in roster:
                    if not isinstance(member, dict):
                        continue
                    status = "alive" if member.get("alive", True) else "down"
                    roster_lines.append(
                        f"- {member.get('id')}: {member.get('role', 'hero')} "
                        f"hp {member.get('hp', '?')}/{member.get('max_hp', '?')} {status}"
                    )
                if roster_lines:
                    roster_text = (
                        "\n\nCurrent player heroes. These are the only valid hero ids "
                        "to address with message actions; use visible_teammates in the "
                        "observation for positions you can currently see:\n"
                        + "\n".join(roster_lines)
                    )
        contract = (
            "You are an OpenEnv ReAct-style DungeonGrid hero policy. Zargon/Warden is controlled "
            "by the environment, not by you. You control only the active hero named in "
            "the observation. You may call dungeongrid_rules to review mechanics before "
            "committing. You may use private plan tools to maintain a medium-term route "
            "and objective plan across turns; this memory is private to the active hero "
            "and is not shared with teammates unless you send a message action. Your "
            "final assistant action must be exactly one dungeongrid_act "
            "tool call. The tool argument must be a JSON object with an "
            "optional intent string and an actions list of structured JSON action objects. "
            "Do not ask for or expect a legal-action list; use the rules and visible board "
            "state, and rely on environment feedback for invalid actions.\n"
            "Common invalid patterns to avoid: repeated blocked movement from the same "
            "position; inspect_tile on far or unseen coordinates; cast spells not listed "
            "as available spell cards in your observation; spending more AP than remains; and putting message actions "
            "after AP-consuming actions when the message is important. Coordinates are only "
            "valid as inspect_tile targets; object/entity actions must target ids from "
            "visible_objects or visible_entities. If an action failed "
            "last turn, change direction, inspect/open nearby features, or end_turn rather "
            "than repeating it. Before committing moves in a corridor, read or update your "
            "private plan so your next action serves the route instead of local oscillation.\n\n"
            f"Compact DungeonGrid rules:\n{rules}"
            f"{roster_text}"
        )
        return f"{base}\n\n{contract}" if base else contract

    def _token_limit_payload(self) -> dict[str, int]:
        field = self.token_limit_field
        if not field:
            lowered = self.model_name.lower()
            field = "max_completion_tokens" if lowered.startswith(("gpt-5", "o")) else "max_tokens"
        return {field: self.max_tokens}

    def _tool_schema(self) -> dict[str, Any]:
        return dungeongrid_act_schema()

    def _private_plan_tool_names(self) -> set[str]:
        return {
            "dungeongrid_private_plan_read",
            "dungeongrid_private_plan_write",
            "dungeongrid_private_plan_append",
            "dungeongrid_private_plan_edit",
        }

    def _private_plan_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "dungeongrid_private_plan_read",
                    "description": (
                        "Read this hero's private medium-term plan. The plan is scratch "
                        "memory for route/objective planning and is not visible to other heroes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dungeongrid_private_plan_write",
                    "description": "Replace this hero's private medium-term plan with concise text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "string",
                                "description": "Concise private plan text, ideally 3-8 bullets.",
                            }
                        },
                        "required": ["plan"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dungeongrid_private_plan_append",
                    "description": "Append a short note to this hero's private medium-term plan.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": "One short operational note to append.",
                            }
                        },
                        "required": ["note"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dungeongrid_private_plan_edit",
                    "description": (
                        "Edit this hero's private plan by replacing exact old text with new text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old": {
                                "type": "string",
                                "description": "Exact text to replace in the current plan.",
                            },
                            "new": {"type": "string", "description": "Replacement text."},
                        },
                        "required": ["old", "new"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _assistant_tool_call_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
        }

    def _rules_tool_result_message(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        topic = str(arguments.get("topic") or "actions").strip() or "actions"
        return self._tool_result_message(
            tool_call,
            {
                "topic": topic,
                "text": dungeongrid_rules(topic),
            },
        )

    def _private_plan_tool_result_message(
        self,
        tool_call: dict[str, Any],
        observation: Observation,
    ) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        self.private_plan_tool_count += 1
        self.private_plan_tool_counts[name] = self.private_plan_tool_counts.get(name, 0) + 1
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        key = self._private_plan_key(observation)
        current = self._private_plans.get(key, "")
        if name == "dungeongrid_private_plan_read":
            result = {"agent_id": key, "plan": current or "(empty)"}
        elif name == "dungeongrid_private_plan_write":
            plan = self._clip_private_plan(str(arguments.get("plan") or ""))
            self._private_plans[key] = plan
            result = {"agent_id": key, "plan": plan or "(empty)", "status": "written"}
        elif name == "dungeongrid_private_plan_append":
            note = str(arguments.get("note") or "").strip()
            updated = current.strip()
            if note:
                updated = f"{updated}\n- {note}" if updated else f"- {note}"
            updated = self._clip_private_plan(updated)
            self._private_plans[key] = updated
            result = {"agent_id": key, "plan": updated or "(empty)", "status": "appended"}
        elif name == "dungeongrid_private_plan_edit":
            old = str(arguments.get("old") or "")
            new = str(arguments.get("new") or "")
            if old and old in current:
                updated = current.replace(old, new, 1)
                status = "edited"
                closest_matches: list[str] = []
            else:
                updated = current
                status = "old_text_not_found"
                closest_matches = self._closest_plan_lines(current, old)
            updated = self._clip_private_plan(updated)
            self._private_plans[key] = updated
            result = {
                "agent_id": key,
                "plan": updated or "(empty)",
                "status": status,
                "closest_matches": closest_matches,
            }
        else:
            result = {"error": f"unknown private plan tool: {name}"}
        return self._tool_result_message(tool_call, result)

    def _private_plan_key(self, observation: Observation) -> str:
        return str(observation.metadata.get("active_agent") or observation.agent_id)

    def _clip_private_plan(self, plan: str) -> str:
        plan = plan.strip()
        max_chars = 1600
        if len(plan) <= max_chars:
            return plan
        return plan[-max_chars:].lstrip()

    def _closest_plan_lines(self, plan: str, old: str) -> list[str]:
        if not old:
            return []
        lines = [line.strip() for line in plan.splitlines() if line.strip()]
        return difflib.get_close_matches(old.strip(), lines, n=3, cutoff=0.35)

    def _tool_result_message(self, tool_call: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": (tool_call.get("function") or {}).get("name", ""),
            "content": json.dumps(result, sort_keys=True),
        }

    def _extract_tool_actions(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        choices = body.get("choices") or []
        if not choices:
            return []
        message = choices[0].get("message") or {}
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "dungeongrid_act":
                continue
            arguments = function.get("arguments") or "{}"
            try:
                payload = json.loads(arguments)
            except json.JSONDecodeError:
                return []
            return _extract_action_objects_from_payload(payload)
        return []

    def _fallback_action_name(self, observation: Observation) -> str:
        return str(next(iter(observation.available_actions), "WAIT"))

    @classmethod
    def from_config(
        cls, package: PolicyPackage, config: dict[str, Any]
    ) -> "DungeonGridReActPolicy":
        model_cfg = config.get("model", {})
        default_api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        api_base = str(model_cfg.get("api_base") or default_api_base)
        api_key = os.getenv("OPENAI_API_KEY", model_cfg.get("api_key", "changeme"))
        temperature = model_cfg.get("temperature", None)
        model_name = str(model_cfg.get("name", package.model or "gpt-5-nano"))
        return cls(
            package=package,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=_dungeongrid_max_tokens(
                model_name,
                model_cfg.get("max_tokens"),
            ),
            timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
            token_limit_field=model_cfg.get("token_limit_field"),
            reasoning_effort=model_cfg.get("reasoning_effort")
            or _default_dungeongrid_reasoning_effort(model_name),
            omit_temperature=bool(model_cfg.get("omit_temperature", False)),
            max_retries=int(model_cfg.get("max_retries", 2)),
            max_tool_rounds=int(model_cfg.get("max_tool_rounds", 6)),
        )


@dataclass
class DungeonGridWardenReActPolicy:
    model_name: str
    api_base: str
    api_key: str
    seed_prompt: str = ""
    name: str = "dungeongrid_warden_react"
    temperature: float | None = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 60.0
    token_limit_field: str | None = None
    reasoning_effort: str | None = None
    omit_temperature: bool = False
    max_retries: int = 2
    max_tool_rounds: int = 3

    def __post_init__(self) -> None:
        self.llm_call_count = 0
        self.llm_usage = _new_llm_usage_totals()
        self.llm_usage_events: list[dict[str, Any]] = []
        self.fallback_count = 0
        self.action_counts: dict[str, int] = {}
        self.decisions: list[dict[str, Any]] = []

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": self._user_prompt(observation),
            },
        ]
        for round_index in range(max(1, self.max_tool_rounds)):
            force_act = round_index == self.max_tool_rounds - 1
            payload = self._chat_payload(
                messages,
                tool_choice={"type": "function", "function": {"name": "dungeongrid_warden_act"}}
                if force_act
                else "auto",
            )
            body = self._post_chat_completions(payload)
            if self._completion_budget_exhausted_without_output(body):
                return self._fallback_action(
                    observation,
                    reason="completion_budget_exhausted_without_output",
                )
            message = (body.get("choices") or [{}])[0].get("message") or {}
            decision = self._extract_warden_decision(body)
            if decision is not None:
                return self._bounded_action(observation, decision)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You did not call a tool. You may call dungeongrid_rules "
                            "for context, but you must finish with dungeongrid_warden_act."
                        ),
                    }
                )
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name")
                if name == "dungeongrid_rules":
                    messages.append(self._rules_tool_result_message(tool_call))
                elif name == "dungeongrid_warden_act":
                    messages.append(
                        self._tool_result_message(
                            tool_call,
                            {"error": "dungeongrid_warden_act requires one valid action object."},
                        )
                    )
                else:
                    messages.append(
                        self._tool_result_message(tool_call, {"error": f"unknown tool: {name}"})
                    )
        return self._fallback_action(observation, reason="no_valid_warden_tool_call")

    def _bounded_action(
        self, observation: dict[str, Any], decision: WardenDecision
    ) -> dict[str, Any]:
        adapter = WardenReActAdapter(lambda _obs: decision, name=self.name)
        action = adapter.act(observation)
        fallback_used = str(action.get("warden_policy", "")).endswith(":fallback")
        if fallback_used:
            self.fallback_count += 1
            action.setdefault("warden_fallback_reason", "outside_bounded_warden_candidates")
        action_type = str(action.get("type") or "unknown")
        self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1
        self.decisions.append(
            {
                "action": dict(action),
                "fallback_used": fallback_used,
                "intent": action.get("warden_intent"),
                "axis_pressure": action.get("warden_axis_pressure"),
                "fairness_check": action.get("warden_fairness_check"),
            }
        )
        return action

    def _fallback_action(self, observation: dict[str, Any], *, reason: str) -> dict[str, Any]:
        self.fallback_count += 1
        action = WardenReActAdapter(lambda _obs: {"type": "__invalid__"}, name=self.name).act(
            observation
        )
        action["warden_fallback_reason"] = reason
        action_type = str(action.get("type") or "unknown")
        self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1
        self.decisions.append({"action": dict(action), "fallback_used": True, "reason": reason})
        return action

    def _extract_warden_decision(self, body: dict[str, Any]) -> WardenDecision | None:
        message = (body.get("choices") or [{}])[0].get("message") or {}
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "dungeongrid_warden_act":
                continue
            try:
                payload = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                return None
            action = payload.get("action")
            if not isinstance(action, dict) or not action.get("type"):
                return None
            try:
                typed_action = DungeonGridAction(**action)
            except ValidationError:
                return None
            return WardenDecision(
                action=typed_action.model_dump(mode="json", exclude_none=True),
                policy=self.name,
                intent=str(payload.get("intent") or "").strip() or None,
                axis_pressure=str(payload.get("axis_pressure") or "").strip() or None,
                fairness_check=str(payload.get("fairness_check") or "").strip() or None,
            )
        return None

    def _chat_payload(self, messages: list[dict[str, Any]], *, tool_choice: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            **self._token_limit_payload(),
            "messages": messages,
            "tools": [dungeongrid_warden_act_schema(), dungeongrid_rules_schema()],
            "tool_choice": tool_choice,
        }
        if not self.omit_temperature and self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: requests.RequestException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.api_base.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                self.llm_call_count += 1
                response.raise_for_status()
                body = response.json()
                _record_llm_usage(
                    self.llm_usage,
                    self.llm_usage_events,
                    model=self.model_name,
                    body=body,
                )
                return body
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
        raise RuntimeError("unreachable DungeonGrid Warden retry state") from last_error

    def _completion_budget_exhausted_without_output(self, body: dict[str, Any]) -> bool:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0
        return (
            choice.get("finish_reason") == "length"
            and not content
            and not tool_calls
            and reasoning_tokens > 0
        )

    def _token_limit_payload(self) -> dict[str, int]:
        field = self.token_limit_field
        if not field:
            lowered = self.model_name.lower()
            field = "max_completion_tokens" if lowered.startswith(("gpt-5", "o")) else "max_tokens"
        return {field: self.max_tokens}

    def _system_prompt(self) -> str:
        base = self.seed_prompt.strip()
        contract = (
            "You are DungeonGrid's private Warden ReAct policy. You are adversarial "
            "but fair: pressure the dungeon's declared MARL/coordination axis, preserve "
            "counterplay, and avoid hidden-information perfect play. You may call "
            "dungeongrid_rules for context. Your final assistant action must be exactly "
            "one dungeongrid_warden_act tool call. Choose one bounded Warden action from "
            "the candidates in the observation; do not invent direct damage, hidden spawns, "
            "monster movement, or state changes outside those candidates. Include an "
            "intent, axis_pressure, and fairness_check so the transcript is reviewable."
        )
        return f"{base}\n\n{contract}" if base else contract

    def _user_prompt(self, observation: dict[str, Any]) -> str:
        selected = {
            "quest_id": observation.get("quest_id"),
            "quest_title": observation.get("quest_title"),
            "round": observation.get("round"),
            "phase": observation.get("phase"),
            "active_agent": observation.get("active_agent"),
            "marl_axis": observation.get("marl_axis"),
            "coordination_type": observation.get("coordination_type"),
            "dread": observation.get("dread"),
            "alert": observation.get("alert"),
            "torch": observation.get("torch"),
            "marl_contract": observation.get("marl_contract"),
            "warden_policy_contract": observation.get("warden_policy_contract"),
            "visible_hero_progress": observation.get("visible_hero_progress"),
            "recent_events": observation.get("recent_events"),
            "recent_party_messages": observation.get("recent_party_messages"),
            "legal_warden_action_candidates": observation.get("legal_actions") or [],
        }
        return (
            "Pick one fair bounded Warden action for this Warden turn.\n\n"
            f"{json.dumps(selected, indent=2, sort_keys=True)}"
        )

    def _rules_tool_result_message(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        topic = str(arguments.get("topic") or "warden").strip() or "warden"
        return self._tool_result_message(
            tool_call,
            {"topic": topic, "text": dungeongrid_rules(topic)},
        )

    def _tool_result_message(self, tool_call: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": (tool_call.get("function") or {}).get("name", ""),
            "content": json.dumps(result, sort_keys=True),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DungeonGridWardenReActPolicy":
        global_model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        warden_cfg = (
            config.get("warden_policy")
            if isinstance(config.get("warden_policy"), dict)
            else {}
        )
        warden_model_cfg = (
            warden_cfg.get("model") if isinstance(warden_cfg.get("model"), dict) else {}
        )
        model_cfg = {**global_model_cfg, **warden_model_cfg}
        model_name = str(model_cfg.get("name") or "gpt-4.1-nano")
        default_api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        api_base = str(model_cfg.get("api_base") or default_api_base)
        api_key = os.getenv("OPENAI_API_KEY", model_cfg.get("api_key", "changeme"))
        temperature = model_cfg.get("temperature", 0.0)
        return cls(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            seed_prompt=str(warden_cfg.get("seed_prompt") or ""),
            name=str(warden_cfg.get("name") or "dungeongrid_warden_react"),
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=_dungeongrid_max_tokens(model_name, model_cfg.get("max_tokens", 512)),
            timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
            token_limit_field=model_cfg.get("token_limit_field"),
            reasoning_effort=model_cfg.get("reasoning_effort")
            or _default_dungeongrid_reasoning_effort(model_name),
            omit_temperature=bool(model_cfg.get("omit_temperature", False)),
            max_retries=int(model_cfg.get("max_retries", 2)),
            max_tool_rounds=int(model_cfg.get("max_tool_rounds", 3)),
        )


def _default_dungeongrid_reasoning_effort(model_name: str) -> str | None:
    lowered = model_name.lower()
    if lowered in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        return "medium"
    return None


def _dungeongrid_max_tokens(model_name: str, configured: Any) -> int:
    value = int(configured) if configured is not None else 768
    lowered = model_name.lower()
    if lowered in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        return max(value, 8192)
    return value


def _extract_action_from_text(content: str) -> str:
    try:
        payload = json.loads(content)
        action = str(payload.get("action", "")).strip()
        if action:
            return action
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"(FETCH_INGREDIENT|PREP_POT|FETCH_DISH|PLATE_SOUP|SERVE_SOUP|WAIT|"
        r"move|open_door|attack_melee|attack_ranged|cast|inspect_tile|inspect_room|"
        r"disarm|interact|use_item|equip_item|give_item|message|guard|end_turn|warden_auto|activate_monster)",
        content,
    )
    if match:
        return match.group(1)
    return "WAIT"


def _extract_actions_from_text(content: str, *, limit: int) -> list[str]:
    actions: list[str] = []
    try:
        payload = json.loads(content)
        raw_actions = payload.get("actions", [])
        if isinstance(raw_actions, list):
            actions = [
                str(action).strip()
                for action in raw_actions
                if str(action).strip()
            ]
    except json.JSONDecodeError:
        actions = []

    if not actions:
        actions = [
            match.group(1)
            for match in re.finditer(
                r"(FETCH_INGREDIENT|PREP_POT|FETCH_DISH|PLATE_SOUP|SERVE_SOUP|WAIT|"
                r"move|open_door|attack_melee|attack_ranged|cast|inspect_tile|inspect_room|"
                r"disarm|interact|use_item|equip_item|give_item|message|guard|end_turn|warden_auto|activate_monster)",
                content,
            )
        ]
    if not actions:
        actions = [_extract_action_from_text(content)]
    while len(actions) < limit:
        actions.append(actions[-1])
    return actions[:limit]


def _extract_action_objects_from_text(content: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(payload, list):
        raw_actions = payload
    elif isinstance(payload, dict):
        raw_actions = payload.get("actions", [])
        if not raw_actions and payload.get("type"):
            raw_actions = [payload]
    else:
        return []
    return _validated_dungeongrid_actions(raw_actions)


def _extract_action_objects_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw_actions = payload.get("actions", [])
        if not raw_actions and payload.get("type"):
            raw_actions = [payload]
    elif isinstance(payload, list):
        raw_actions = payload
    else:
        return []
    return _validated_dungeongrid_actions(raw_actions)


def _validated_dungeongrid_actions(raw_actions: Any) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not isinstance(raw_actions, list):
        return actions
    for action in raw_actions:
        if not isinstance(action, dict) or not action.get("type"):
            continue
        try:
            typed_action = DungeonGridAction(**action)
        except ValidationError:
            continue
        actions.append(typed_action.model_dump(mode="json", exclude_none=True))
    return actions


def _canonicalize_dungeongrid_actions(
    actions: list[dict[str, Any]], legal: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for action in actions:
        match = _find_matching_dungeongrid_action(action, legal)
        resolved = dict(match or action)
        if resolved.get("type") in {"message", "give_item"} and isinstance(action.get("payload"), dict):
            resolved["payload"] = dict(action["payload"])
        canonical.append(resolved)
    return canonical


def _find_matching_dungeongrid_action(
    requested: dict[str, Any], legal: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for action in legal:
        if not _dungeongrid_action_matches(requested, action):
            continue
        return action
    return None


def _dungeongrid_action_matches(requested: dict[str, Any], legal: dict[str, Any]) -> bool:
    if requested.get("type") != legal.get("type"):
        return False
    for key in ("direction", "target", "payload"):
        if key in legal and key not in requested:
            return False
        if key in legal and requested.get(key) != legal.get(key):
            return False
    return True


def make_seed_package(
    name: str, backend: str, prompt: str, model_name: str | None = None
) -> PolicyPackage:
    return PolicyPackage(
        name=name,
        backend=backend,
        system_prompt=prompt.strip(),
        behavior_flags=extract_behavior_flags(prompt),
        model=model_name,
    )
