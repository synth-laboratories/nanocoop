from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from nanocoop.constants import MOCK_ACTIONS
from nanocoop.schema import EpisodeTrace, Observation, StepRecord


@dataclass
class MockState:
    layout: str
    seed: int
    recipe: str
    pot_side: str
    convention: str
    ingredient_ready: bool = False
    pot_ready: bool = False
    dish_ready: bool = False
    plated: bool = False
    delivered: bool = False
    shared_recipe_known: bool = False
    shared_pot_known: bool = False
    step_index: int = 0
    last_event: str | None = None
    last_actions: tuple[str | None, str | None] = (None, None)
    recent_events: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.recent_events is None:
            self.recent_events = []


class MockOvercookedV2Backend:
    def __init__(self, *, max_steps: int = 8, stochasticity: float = 0.12) -> None:
        self.max_steps = max_steps
        self.stochasticity = stochasticity

    def rollout(
        self,
        *,
        focal_policy,
        partner_policy,
        layout: str,
        seed: int,
        partner_name: str,
        mode: str = "cross_play",
    ) -> EpisodeTrace:
        rng = random.Random((hash((layout, seed, partner_name, mode)) & 0xFFFFFFFF))
        state = self._reset_state(layout=layout, seed=seed, rng=rng)
        steps: list[StepRecord] = []
        total_reward = 0.0

        while state.step_index < self.max_steps and not state.delivered:
            obs0 = self._observe(state, "agent_0")
            obs1 = self._observe(state, "agent_1")
            action0 = focal_policy.act(obs0)
            action1 = partner_policy.act(obs1)
            reward, event = self._step(state, action0, action1, rng=rng)
            total_reward += reward
            steps.append(
                StepRecord(
                    step_index=state.step_index,
                    action_agent_0=action0,
                    action_agent_1=action1,
                    reward=reward,
                    event=event,
                    focal_observation=obs0,
                )
            )

        return EpisodeTrace(
            layout=layout,
            seed=seed,
            partner_name=partner_name,
            total_reward=round(total_reward, 4),
            success=state.delivered,
            steps=steps,
            metadata={
                "mode": mode,
                "recipe": state.recipe,
                "pot_side": state.pot_side,
                "convention": state.convention,
                "max_steps": self.max_steps,
            },
        )

    def _reset_state(self, *, layout: str, seed: int, rng: random.Random) -> MockState:
        recipe = rng.choice(["onion", "tomato"])
        pot_side = rng.choice(["left", "right"])
        convention = rng.choice(["focal_fetches", "partner_fetches"])
        if "grounded" in layout:
            convention = "focal_fetches"
        if "wide" in layout:
            convention = "partner_fetches"
        return MockState(
            layout=layout,
            seed=seed,
            recipe=recipe,
            pot_side=pot_side,
            convention=convention,
        )

    def _observe(self, state: MockState, agent_id: str) -> Observation:
        private_recipe = state.recipe if agent_id == "agent_0" else None
        private_pot = state.pot_side if agent_id == "agent_1" else None
        last_partner_action = state.last_actions[1] if agent_id == "agent_0" else state.last_actions[0]
        hint = state.convention if "test_time" in state.layout and agent_id == "agent_1" else None
        return Observation(
            agent_id=agent_id,
            layout=state.layout,
            step_index=state.step_index,
            max_steps=self.max_steps,
            private_recipe=private_recipe,
            private_pot=private_pot,
            shared_recipe_known=state.shared_recipe_known,
            shared_pot_known=state.shared_pot_known,
            ingredient_ready=state.ingredient_ready,
            pot_ready=state.pot_ready,
            dish_ready=state.dish_ready,
            plated=state.plated,
            delivered=state.delivered,
            last_partner_action=last_partner_action,
            last_joint_event=state.last_event,
            available_actions=MOCK_ACTIONS,
            convention_hint=hint,
            recent_events=tuple(state.recent_events[-4:]),
        )

    def _step(
        self,
        state: MockState,
        action0: str,
        action1: str,
        *,
        rng: random.Random,
    ) -> tuple[float, str]:
        reward = -0.03
        event_parts: list[str] = []

        reward_delta_0, event_0 = self._apply_action(state, action0, actor="agent_0", rng=rng)
        reward += reward_delta_0
        if event_0:
            event_parts.append(f"a0:{event_0}")

        reward_delta_1, event_1 = self._apply_action(state, action1, actor="agent_1", rng=rng)
        reward += reward_delta_1
        if event_1:
            event_parts.append(f"a1:{event_1}")

        if (
            {action0, action1} == {"FETCH_INGREDIENT", "PREP_POT"}
            and not state.plated
        ):
            reward += 0.18
            event_parts.append("parallel_progress")

        if action0 == action1 and action0 in {"FETCH_DISH", "FETCH_INGREDIENT", "PREP_POT"}:
            reward -= 0.12
            event_parts.append("duplicate_work")

        if "test_time" in state.layout:
            if state.convention == "partner_fetches" and action0 == "FETCH_DISH":
                reward += 0.08
                event_parts.append("adapted_to_partner_style")
            elif state.convention == "partner_fetches" and action0 == "FETCH_INGREDIENT" and state.step_index < 2:
                reward -= 0.08
                event_parts.append("missed_partner_style")
            elif state.convention == "focal_fetches" and action0 == "FETCH_INGREDIENT":
                reward += 0.06
                event_parts.append("matched_focal_style")

        if "demo_cook" in state.layout and state.dish_ready and not state.pot_ready:
            reward -= 0.04
            event_parts.append("dish_too_early")

        reward = round(reward, 4)
        state.step_index += 1
        state.last_actions = (action0, action1)
        state.last_event = ", ".join(event_parts) if event_parts else "quiet_step"
        state.recent_events.append(state.last_event)
        return reward, state.last_event

    def _apply_action(
        self,
        state: MockState,
        action: str,
        *,
        actor: str,
        rng: random.Random,
    ) -> tuple[float, str | None]:
        if action not in MOCK_ACTIONS:
            return -0.2, "invalid_action"

        if action == "WAIT":
            return -0.01, "wait"

        if action == "SHARE_RECIPE":
            if actor == "agent_0" and not state.shared_recipe_known:
                state.shared_recipe_known = True
                return 0.16, "shared_recipe"
            return -0.05, "wasted_share_recipe"

        if action == "SHARE_POT":
            if actor == "agent_1" and not state.shared_pot_known:
                state.shared_pot_known = True
                return 0.16, "shared_pot"
            return -0.05, "wasted_share_pot"

        if action == "FETCH_INGREDIENT":
            if state.ingredient_ready:
                return -0.05, "ingredient_already_ready"
            if rng.random() < self.stochasticity * (1.3 if "wide" in state.layout else 1.0):
                return -0.08, "ingredient_delay"
            if actor == "agent_1" and state.convention == "focal_fetches" and state.step_index < 2:
                return 0.1, "partner_helped_fetch"
            state.ingredient_ready = True
            return 0.34, "ingredient_ready"

        if action == "PREP_POT":
            if state.pot_ready:
                return -0.05, "pot_already_ready"
            if rng.random() < self.stochasticity:
                return -0.08, "pot_jam"
            state.pot_ready = True
            return 0.28, "pot_ready"

        if action == "FETCH_DISH":
            if state.dish_ready:
                return -0.05, "dish_already_ready"
            if rng.random() < self.stochasticity * 0.5:
                return -0.06, "dish_delay"
            state.dish_ready = True
            return 0.22, "dish_ready"

        if action == "PLATE_SOUP":
            if state.plated:
                return -0.05, "already_plated"
            if state.ingredient_ready and state.pot_ready and state.dish_ready:
                state.plated = True
                return 0.48, "soup_plated"
            return -0.12, "plate_failed"

        if action == "SERVE_SOUP":
            if state.delivered:
                return -0.05, "already_served"
            if state.plated:
                state.delivered = True
                bonus = 1.1 if "test_time" in state.layout else 1.0
                return bonus, "soup_served"
            return -0.14, "serve_failed"

        return -0.1, "unknown_action"
