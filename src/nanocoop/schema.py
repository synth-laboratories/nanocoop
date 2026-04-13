from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
