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
        if (
            "share_hidden_info_early" in self._flags
            and observation.step_index <= 1
            and not observation.shared_recipe_known
            and observation.private_recipe is not None
        ):
            return "SHARE_RECIPE"
        if (
            "share_hidden_info_early" in self._flags
            and observation.step_index <= 1
            and not observation.shared_pot_known
            and observation.private_pot is not None
        ):
            return "SHARE_POT"

        if (
            "infer_partner_convention" in self._flags
            and observation.last_partner_action == "FETCH_DISH"
            and not observation.ingredient_ready
        ):
            return "FETCH_INGREDIENT"
        if (
            "infer_partner_convention" in self._flags
            and observation.last_partner_action == "PREP_POT"
            and not observation.ingredient_ready
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
            if not observation.ingredient_ready:
                return "FETCH_INGREDIENT"
            if not observation.pot_ready:
                return "PREP_POT"

        if not observation.ingredient_ready:
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
        if not observation.shared_recipe_known and observation.private_recipe is not None:
            return "SHARE_RECIPE"
        if not observation.shared_pot_known and observation.private_pot is not None:
            return "SHARE_POT"
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
            and not observation.ingredient_ready
        ):
            return "FETCH_INGREDIENT"
        if observation.last_partner_action == "PREP_POT" and not observation.ingredient_ready:
            return "FETCH_INGREDIENT"

        if not observation.ingredient_ready:
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
    timeout_seconds: float = 30.0

    def act(self, observation: Observation) -> str:
        prompt = observation.to_prompt()
        examples_text = render_fewshot_examples(self.package.fewshot_examples[:4])
        system_prompt = self.package.system_prompt.strip()
        if examples_text:
            system_prompt = f"{system_prompt}\n\nFew-shot examples:\n{examples_text}"

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
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
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return _extract_action_from_text(content)

    @classmethod
    def from_config(
        cls, package: PolicyPackage, config: dict[str, Any]
    ) -> "RemoteChatPolicy":
        model_cfg = config.get("model", {})
        api_base = os.getenv(
            "OPENAI_API_BASE", model_cfg.get("api_base", "http://127.0.0.1:8000/v1")
        )
        api_key = os.getenv("OPENAI_API_KEY", model_cfg.get("api_key", "changeme"))
        return cls(
            package=package,
            model_name=str(model_cfg.get("name", package.model or "unknown-model")),
            api_base=api_base,
            api_key=api_key,
            temperature=float(model_cfg.get("temperature", 0.0)),
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
        r"(SHARE_RECIPE|SHARE_POT|FETCH_INGREDIENT|PREP_POT|"
        r"FETCH_DISH|PLATE_SOUP|SERVE_SOUP|WAIT)",
        content,
    )
    if match:
        return match.group(1)
    return "WAIT"


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
