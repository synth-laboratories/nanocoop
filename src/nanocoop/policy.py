from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from nanocoop.constants import COOP_ACTIONS
from nanocoop.prompts import extract_behavior_flags, render_fewshot_examples
from nanocoop.schema import Observation, PolicyPackage


class Policy(Protocol):
    def act(self, observation: Observation) -> str:
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
        if self.plan_horizon > 1:
            return self._act_from_plan(observation)
        return self._act_once(observation)

    def __post_init__(self) -> None:
        self._planned_actions: list[str] = []
        self._last_plan_step = -1
        self._plan_state_key: tuple[Any, ...] | None = None
        self.llm_call_count = 0

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
                        f"{list(COOP_ACTIONS)}."
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
                        f"{list(COOP_ACTIONS)}. Return exactly "
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
        content = body["choices"][0]["message"]["content"]
        return _extract_actions_from_text(content, limit=self.plan_horizon)

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


def _extract_action_from_text(content: str) -> str:
    try:
        payload = json.loads(content)
        action = str(payload.get("action", "")).strip()
        if action in COOP_ACTIONS:
            return action
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"(FETCH_INGREDIENT|PREP_POT|FETCH_DISH|PLATE_SOUP|SERVE_SOUP|WAIT)",
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
                if str(action).strip() in COOP_ACTIONS
            ]
    except json.JSONDecodeError:
        actions = []

    if not actions:
        actions = [
            match.group(1)
            for match in re.finditer(
                r"(FETCH_INGREDIENT|PREP_POT|FETCH_DISH|PLATE_SOUP|SERVE_SOUP|WAIT)",
                content,
            )
        ]
    if not actions:
        actions = [_extract_action_from_text(content)]
    while len(actions) < limit:
        actions.append(actions[-1])
    return actions[:limit]


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
