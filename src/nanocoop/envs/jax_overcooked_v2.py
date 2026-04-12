from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nanocoop.constants import COOP_ACTIONS
from nanocoop.schema import EpisodeTrace, Observation, StepRecord


MACRO_ACTIONS = set(COOP_ACTIONS)
PRIMITIVE_ACTIONS = {
    "RIGHT",
    "DOWN",
    "LEFT",
    "UP",
    "STAY",
    "INTERACT",
    "right",
    "down",
    "left",
    "up",
    "stay",
    "interact",
}


@dataclass
class JaxOvercookedV2Backend:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            import jax
            from jaxmarl.environments.overcooked_v2.common import (
                Actions,
                Direction,
                DynamicObject,
                StaticObject,
            )
            from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
        except Exception as exc:  # pragma: no cover - dependency-gated path
            raise RuntimeError(
                "The JAX OvercookedV2 backend requires the optional 'overcookedv2' "
                "dependencies. Install with: pip install -e '.[overcookedv2]'"
            ) from exc
        self.jax = jax
        self.Actions = Actions
        self.Direction = Direction
        self.DynamicObject = DynamicObject
        self.StaticObject = StaticObject
        self.OvercookedV2 = OvercookedV2
        env_cfg = self.config.get("env", {})
        self.max_steps = int(env_cfg.get("max_steps", 400))
        self.negative_rewards = bool(env_cfg.get("negative_rewards", False))
        self.random_reset = bool(env_cfg.get("random_reset", False))
        self.random_agent_positions = bool(env_cfg.get("random_agent_positions", False))
        self.start_cooking_interaction = bool(
            env_cfg.get("start_cooking_interaction", False)
        )
        self.sample_recipe_on_delivery = bool(
            env_cfg.get("sample_recipe_on_delivery", False)
        )
        self._env_cache: dict[str, Any] = {}

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
        env = self._env(layout)
        key = self.jax.random.PRNGKey(int(seed))
        key, reset_key = self.jax.random.split(key)
        _, state = env.reset(reset_key)

        steps: list[StepRecord] = []
        total_reward = 0.0
        delivered = False
        last_actions: tuple[str | None, str | None] = (None, None)
        last_event: str | None = None
        recent_events: list[str] = []

        for step_index in range(self.max_steps):
            obs0 = self._observation(
                env=env,
                state=state,
                agent_index=0,
                layout=layout,
                step_index=step_index,
                last_partner_action=last_actions[1],
                last_event=last_event,
                recent_events=recent_events,
            )
            obs1 = self._observation(
                env=env,
                state=state,
                agent_index=1,
                layout=layout,
                step_index=step_index,
                last_partner_action=last_actions[0],
                last_event=last_event,
                recent_events=recent_events,
            )
            action0_name = str(focal_policy.act(obs0))
            action1_name = str(partner_policy.act(obs1))
            primitive0 = self._primitive_action(
                state, agent_index=0, action_name=action0_name
            )
            primitive1 = self._primitive_action(
                state, agent_index=1, action_name=action1_name
            )

            key, step_key = self.jax.random.split(key)
            _, state, rewards, dones, info = env.step_env(
                step_key,
                state,
                {"agent_0": primitive0, "agent_1": primitive1},
            )
            reward = float(np.asarray(rewards["agent_0"]))
            total_reward += reward
            delivered = delivered or self._delivered(state=state, reward=reward)
            event = self._event_name(
                action0=action0_name,
                action1=action1_name,
                primitive0=int(np.asarray(primitive0)),
                primitive1=int(np.asarray(primitive1)),
                reward=reward,
                info=info,
                delivered=delivered,
            )
            recent_events.append(event)
            steps.append(
                StepRecord(
                    step_index=step_index,
                    action_agent_0=action0_name,
                    action_agent_1=action1_name,
                    reward=round(reward, 4),
                    event=event,
                    focal_observation=obs0,
                )
            )
            last_actions = (action0_name, action1_name)
            last_event = event
            if bool(np.asarray(dones["__all__"])):
                break

        return EpisodeTrace(
            layout=layout,
            seed=int(seed),
            partner_name=partner_name,
            total_reward=round(total_reward, 4),
            success=delivered,
            steps=steps,
            metadata={
                "mode": mode,
                "backend": "jax_overcooked_v2",
                "max_steps": self.max_steps,
                "primitive_action_space": [
                    "right",
                    "down",
                    "left",
                    "up",
                    "stay",
                    "interact",
                ],
            },
        )

    def _env(self, layout: str):
        if layout not in self._env_cache:
            self._env_cache[layout] = self.OvercookedV2(
                layout=layout,
                max_steps=self.max_steps,
                negative_rewards=self.negative_rewards,
                random_reset=self.random_reset,
                random_agent_positions=self.random_agent_positions,
                start_cooking_interaction=self.start_cooking_interaction,
                sample_recipe_on_delivery=self.sample_recipe_on_delivery,
            )
        return self._env_cache[layout]

    def _observation(
        self,
        *,
        env,
        state,
        agent_index: int,
        layout: str,
        step_index: int,
        last_partner_action: str | None,
        last_event: str | None,
        recent_events: list[str],
    ) -> Observation:
        grid = np.asarray(state.grid)
        static = grid[:, :, 0]
        dynamic = grid[:, :, 1]
        extra = grid[:, :, 2]
        recipe = self._recipe_text(int(np.asarray(state.recipe)))
        inventory = int(np.asarray(state.agents.inventory[agent_index]))
        ingredient_ready = bool(np.any(dynamic & self._ingredient_mask()))
        pot_ready = bool(
            np.any(
                (static == int(self.StaticObject.POT))
                & (
                    (dynamic & int(self.DynamicObject.COOKED))
                    == int(self.DynamicObject.COOKED)
                )
            )
        )
        dish_ready = bool(
            np.any(dynamic & int(self.DynamicObject.PLATE))
            or inventory == int(self.DynamicObject.PLATE)
        )
        plated = bool(
            (inventory & int(self.DynamicObject.PLATE))
            and (inventory & int(self.DynamicObject.COOKED))
        )
        delivered = self._delivered(state=state, reward=0.0)
        private_pot = self._nearest_pot_summary(
            static=static,
            dynamic=dynamic,
            extra=extra,
            agent_index=agent_index,
            state=state,
        )
        return Observation(
            agent_id=f"agent_{agent_index}",
            layout=layout,
            step_index=step_index,
            max_steps=self.max_steps,
            private_recipe=recipe,
            private_pot=private_pot,
            shared_recipe_known=True,
            shared_pot_known=True,
            ingredient_ready=ingredient_ready,
            pot_ready=pot_ready,
            dish_ready=dish_ready,
            plated=plated,
            delivered=delivered,
            last_partner_action=last_partner_action,
            last_joint_event=last_event,
            available_actions=tuple(
                sorted(MACRO_ACTIONS | {a.upper() for a in PRIMITIVE_ACTIONS})
            ),
            recent_events=tuple(recent_events[-4:]),
            metadata={
                "agent_position": self._agent_pos(state, agent_index),
                "agent_direction": int(np.asarray(state.agents.dir[agent_index])),
                "inventory": inventory,
            },
        )

    def _primitive_action(self, state, *, agent_index: int, action_name: str):
        action = action_name.strip()
        primitive = self._primitive_action_value(action)
        if primitive is not None:
            return primitive

        target_kind = {
            "FETCH_INGREDIENT": "ingredient_pile",
            "PREP_POT": "pot",
            "FETCH_DISH": "plate_pile",
            "PLATE_SOUP": "pot",
            "SERVE_SOUP": "goal",
        }.get(action)
        if target_kind is None:
            return self.Actions.stay
        return self._plan_toward_target(
            state, agent_index=agent_index, target_kind=target_kind
        )

    def _primitive_action_value(self, action_name: str):
        normalized = action_name.strip().lower()
        mapping = {
            "right": self.Actions.right,
            "down": self.Actions.down,
            "left": self.Actions.left,
            "up": self.Actions.up,
            "stay": self.Actions.stay,
            "wait": self.Actions.stay,
            "interact": self.Actions.interact,
        }
        return mapping.get(normalized)

    def _plan_toward_target(self, state, *, agent_index: int, target_kind: str):
        static = np.asarray(state.grid[:, :, 0])
        dynamic = np.asarray(state.grid[:, :, 1])
        agent_pos = self._agent_pos(state, agent_index)
        targets = self._target_positions(
            static=static, dynamic=dynamic, target_kind=target_kind
        )
        if not targets:
            return self.Actions.stay
        target = min(
            targets,
            key=lambda pos: abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1]),
        )
        delta = (target[0] - agent_pos[0], target[1] - agent_pos[1])
        if abs(delta[0]) + abs(delta[1]) == 1:
            desired_direction = self._direction_for_delta(delta)
            current_direction = int(np.asarray(state.agents.dir[agent_index]))
            if desired_direction is not None and current_direction == int(desired_direction):
                return self.Actions.interact
            return self._action_for_direction(desired_direction)

        for direction, move_delta in self._ranked_directions(delta):
            next_pos = (agent_pos[0] + move_delta[0], agent_pos[1] + move_delta[1])
            if self._is_open(static, next_pos):
                return self._action_for_direction(direction)
        return self.Actions.stay

    def _target_positions(
        self, *, static: np.ndarray, dynamic: np.ndarray, target_kind: str
    ) -> list[tuple[int, int]]:
        del dynamic
        if target_kind == "ingredient_pile":
            mask = static >= int(self.StaticObject.INGREDIENT_PILE_BASE)
        elif target_kind == "plate_pile":
            mask = static == int(self.StaticObject.PLATE_PILE)
        elif target_kind == "pot":
            mask = static == int(self.StaticObject.POT)
        elif target_kind == "goal":
            mask = static == int(self.StaticObject.GOAL)
        else:
            mask = np.zeros_like(static, dtype=bool)
        ys, xs = np.where(mask)
        return [(int(x), int(y)) for y, x in zip(ys, xs)]

    def _agent_pos(self, state, agent_index: int) -> tuple[int, int]:
        return (
            int(np.asarray(state.agents.pos.x[agent_index])),
            int(np.asarray(state.agents.pos.y[agent_index])),
        )

    def _is_open(self, static: np.ndarray, pos: tuple[int, int]) -> bool:
        x, y = pos
        if y < 0 or x < 0 or y >= static.shape[0] or x >= static.shape[1]:
            return False
        return int(static[y, x]) == int(self.StaticObject.EMPTY)

    def _ranked_directions(self, delta: tuple[int, int]):
        dx, dy = delta
        candidates = []
        if abs(dx) >= abs(dy):
            if dx > 0:
                candidates.append((self.Direction.RIGHT, (1, 0)))
            elif dx < 0:
                candidates.append((self.Direction.LEFT, (-1, 0)))
            if dy > 0:
                candidates.append((self.Direction.DOWN, (0, 1)))
            elif dy < 0:
                candidates.append((self.Direction.UP, (0, -1)))
        else:
            if dy > 0:
                candidates.append((self.Direction.DOWN, (0, 1)))
            elif dy < 0:
                candidates.append((self.Direction.UP, (0, -1)))
            if dx > 0:
                candidates.append((self.Direction.RIGHT, (1, 0)))
            elif dx < 0:
                candidates.append((self.Direction.LEFT, (-1, 0)))
        candidates.extend(
            [
                (self.Direction.RIGHT, (1, 0)),
                (self.Direction.DOWN, (0, 1)),
                (self.Direction.LEFT, (-1, 0)),
                (self.Direction.UP, (0, -1)),
            ]
        )
        seen = set()
        ranked = []
        for direction, move_delta in candidates:
            if int(direction) in seen:
                continue
            seen.add(int(direction))
            ranked.append((direction, move_delta))
        return ranked

    def _direction_for_delta(self, delta: tuple[int, int]):
        return {
            (1, 0): self.Direction.RIGHT,
            (-1, 0): self.Direction.LEFT,
            (0, 1): self.Direction.DOWN,
            (0, -1): self.Direction.UP,
        }.get(delta)

    def _action_for_direction(self, direction):
        if direction == self.Direction.RIGHT:
            return self.Actions.right
        if direction == self.Direction.DOWN:
            return self.Actions.down
        if direction == self.Direction.LEFT:
            return self.Actions.left
        if direction == self.Direction.UP:
            return self.Actions.up
        return self.Actions.stay

    def _ingredient_mask(self) -> int:
        mask = 0
        num_ingredients = int(self.config.get("env", {}).get("num_ingredients", 4))
        for idx in range(num_ingredients):
            mask |= int(self.DynamicObject.ingredient(idx))
        return mask

    def _recipe_text(self, recipe_encoding: int) -> str:
        try:
            ingredients = self.DynamicObject.get_ingredient_idx_list(recipe_encoding)
        except Exception:
            ingredients = []
        return ",".join(f"ingredient_{idx}" for idx in ingredients) or "unknown"

    def _nearest_pot_summary(
        self,
        *,
        static: np.ndarray,
        dynamic: np.ndarray,
        extra: np.ndarray,
        agent_index: int,
        state,
    ) -> str:
        agent_pos = self._agent_pos(state, agent_index)
        pots = self._target_positions(static=static, dynamic=dynamic, target_kind="pot")
        if not pots:
            return "none"
        x, y = min(pots, key=lambda pos: abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1]))
        pot_dynamic = int(dynamic[y, x])
        cooked = bool(pot_dynamic & int(self.DynamicObject.COOKED))
        timer = int(extra[y, x])
        count = self._ingredient_count(pot_dynamic)
        return f"nearest_pot ingredients={count} cooked={cooked} timer={timer}"

    def _ingredient_count(self, dynamic_value: int) -> int:
        obj = int(dynamic_value) >> 2
        count = 0
        while obj > 0:
            count += obj & 0x3
            obj >>= 2
        return count

    def _delivered(self, *, state, reward: float) -> bool:
        if bool(np.asarray(getattr(state, "new_correct_delivery", False))):
            return True
        return reward >= 10.0

    def _event_name(
        self,
        *,
        action0: str,
        action1: str,
        primitive0: int,
        primitive1: int,
        reward: float,
        info: dict[str, Any],
        delivered: bool,
    ) -> str:
        shaped = info.get("shaped_reward", {}) if isinstance(info, dict) else {}
        shaped0 = (
            float(np.asarray(shaped.get("agent_0", 0.0)))
            if isinstance(shaped, dict)
            else 0.0
        )
        if delivered:
            return "soup_delivered"
        if reward > 0 or shaped0 > 0:
            return f"progress reward={round(reward, 4)} shaped={round(shaped0, 4)}"
        if primitive0 == primitive1 == int(self.Actions.interact):
            return "both_interacted"
        return (
            f"a0:{action0}->primitive_{primitive0}, "
            f"a1:{action1}->primitive_{primitive1}"
        )
