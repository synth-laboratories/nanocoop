from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
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
                "dependencies. Install with: uv sync --extra overcookedv2"
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
        self.policy_decision_interval = max(
            1, int(env_cfg.get("policy_decision_interval", 1))
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
        capture_states: bool = False,
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
        state_sequence = [state] if capture_states else None
        cached_policy_actions: tuple[str | None, str | None] = (None, None)

        for step_index in range(self.max_steps):
            obs0 = self._observation(
                env=env,
                state=state,
                agent_index=0,
                layout=layout,
                partner_name=partner_name,
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
                partner_name=partner_name,
                step_index=step_index,
                last_partner_action=last_actions[0],
                last_event=last_event,
                recent_events=recent_events,
            )
            should_refresh_policy = (
                step_index == 0
                or step_index % self.policy_decision_interval == 0
                or last_event == "soup_delivered"
            )
            if should_refresh_policy or cached_policy_actions[0] is None:
                action0_name = str(focal_policy.act(obs0))
            else:
                action0_name = str(cached_policy_actions[0])
            action1_name = str(partner_policy.act(obs1))
            cached_policy_actions = (action0_name, action1_name)
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
            if state_sequence is not None:
                state_sequence.append(state)
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
            if delivered or bool(np.asarray(dones["__all__"])):
                break

        metadata: dict[str, Any] = {
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
            "focal_llm_call_count": int(getattr(focal_policy, "llm_call_count", 0) or 0),
        }
        if state_sequence is not None:
            metadata["state_sequence"] = state_sequence
        return EpisodeTrace(
            layout=layout,
            seed=int(seed),
            partner_name=partner_name,
            total_reward=round(total_reward, 4),
            success=delivered,
            steps=steps,
            metadata=metadata,
        )

    def write_rollout_gif(self, trace: EpisodeTrace, path: str | Path) -> Path:
        state_sequence = trace.metadata.get("state_sequence")
        if not state_sequence:
            raise ValueError("rollout trace does not include captured states")
        import jax
        import jax.numpy as jnp
        from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

        state_seq = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *state_sequence)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        OvercookedV2Visualizer().animate(state_seq, filename=str(out))
        return out

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
        partner_name: str,
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
            convention_hint=self._convention_hint(layout=layout, partner_name=partner_name),
            recent_events=tuple(recent_events[-4:]),
            metadata={
                "partner_name": partner_name,
                "agent_position": self._agent_pos(state, agent_index),
                "partner_position": self._agent_pos(state, 1 - agent_index),
                "agent_direction": int(np.asarray(state.agents.dir[agent_index])),
                "inventory": self._inventory_text(inventory),
                "can_reach_ingredient": self._can_reach_target(
                    state, agent_index=agent_index, target_kind="ingredient_pile"
                ),
                "can_reach_pot": self._can_reach_target(
                    state, agent_index=agent_index, target_kind="pot"
                ),
                "can_reach_plate": self._can_reach_target(
                    state, agent_index=agent_index, target_kind="plate_pile"
                ),
                "can_reach_goal": self._can_reach_target(
                    state, agent_index=agent_index, target_kind="goal"
                ),
                "pot_ingredient_count": self._max_pot_ingredient_count(
                    static=static, dynamic=dynamic
                ),
                "pot_full": self._max_pot_ingredient_count(
                    static=static, dynamic=dynamic
                )
                >= 3,
                "nearby": self._nearby_static_summary(
                    static=static, state=state, agent_index=agent_index
                ),
                "pots": self._pot_summaries(static=static, dynamic=dynamic, extra=extra),
                "loose_objects": self._loose_object_summary(dynamic=dynamic),
            },
        )

    def _convention_hint(self, *, layout: str, partner_name: str) -> str:
        layout_hint = "wide layout: commit to one role long enough to finish it." if "wide" in layout else "compact layout: adapt quickly and avoid duplicate work."
        partner_hint = {
            "courier": "partner courier often handles dish delivery support, so you can spend more time on pot progress.",
            "potter": "partner potter tends to fill pots, so bias toward dish pickup, plating, and delivery.",
            "handoff": "partner handoff mirrors recent work, so choose a clear complementary role and stick to it.",
            "noisy": "partner noisy is unreliable, so prefer self-sufficient plans that can still finish the soup.",
        }.get(partner_name, "complement your partner and avoid duplicate work.")
        return f"{layout_hint} {partner_hint}"

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
        recipe_ingredients = self._recipe_ingredient_indices(state)
        targets = self._target_positions(
            static=static,
            dynamic=dynamic,
            target_kind=target_kind,
            recipe_ingredients=recipe_ingredients,
        )
        if not targets:
            return self.Actions.stay
        route = self._route_to_interaction_target(
            static=static,
            agent_pos=agent_pos,
            current_direction=int(np.asarray(state.agents.dir[agent_index])),
            targets=targets,
        )
        if route is None:
            return self.Actions.stay
        if route == "interact":
            return self.Actions.interact
        return self._action_for_direction(route)

    def _can_reach_target(self, state, *, agent_index: int, target_kind: str) -> bool:
        static = np.asarray(state.grid[:, :, 0])
        dynamic = np.asarray(state.grid[:, :, 1])
        targets = self._target_positions(
            static=static,
            dynamic=dynamic,
            target_kind=target_kind,
            recipe_ingredients=self._recipe_ingredient_indices(state),
        )
        return (
            self._route_to_interaction_target(
                static=static,
                agent_pos=self._agent_pos(state, agent_index),
                current_direction=int(np.asarray(state.agents.dir[agent_index])),
                targets=targets,
            )
            is not None
        )

    def _route_to_interaction_target(
        self,
        *,
        static: np.ndarray,
        agent_pos: tuple[int, int],
        current_direction: int,
        targets: list[tuple[int, int]],
    ):
        target_set = set(targets)
        for direction, move_delta in self._ranked_directions((0, 0)):
            adjacent = (agent_pos[0] + move_delta[0], agent_pos[1] + move_delta[1])
            if adjacent in target_set:
                if current_direction == int(direction):
                    return "interact"
                return direction

        goals = set()
        goal_to_direction = {}
        for target in targets:
            for direction, move_delta in self._ranked_directions((0, 0)):
                goal = (target[0] - move_delta[0], target[1] - move_delta[1])
                if self._is_open(static, goal):
                    goals.add(goal)
                    goal_to_direction[goal] = direction
        if not goals:
            return None

        queue = deque([(agent_pos, [])])
        seen = {agent_pos}
        while queue:
            pos, path = queue.popleft()
            if pos in goals:
                if path:
                    return path[0]
                return goal_to_direction[pos]
            for direction, move_delta in self._ranked_directions((0, 0)):
                nxt = (pos[0] + move_delta[0], pos[1] + move_delta[1])
                if nxt in seen or not self._is_open(static, nxt):
                    continue
                seen.add(nxt)
                queue.append((nxt, [*path, direction]))
        return None

    def _target_positions(
        self,
        *,
        static: np.ndarray,
        dynamic: np.ndarray,
        target_kind: str,
        recipe_ingredients: list[int] | None = None,
    ) -> list[tuple[int, int]]:
        del dynamic
        if target_kind == "ingredient_pile":
            if recipe_ingredients:
                mask = np.zeros_like(static, dtype=bool)
                for ingredient_idx in recipe_ingredients:
                    mask |= static == int(self.StaticObject.INGREDIENT_PILE_BASE) + ingredient_idx
            else:
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

    def _inventory_text(self, inventory: int) -> str:
        parts: list[str] = []
        if inventory & int(self.DynamicObject.PLATE):
            parts.append("plate")
        if inventory & int(self.DynamicObject.COOKED):
            parts.append("cooked_soup")
        parts.extend(self._ingredient_names(inventory))
        return ",".join(parts) if parts else "empty"

    def _recipe_text(self, recipe_encoding: int) -> str:
        try:
            ingredients = self.DynamicObject.get_ingredient_idx_list(
                np.asarray(recipe_encoding)
            )
        except Exception:
            ingredients = []
        return ",".join(f"ingredient_{idx}" for idx in ingredients) or "unknown"

    def _recipe_ingredient_indices(self, state) -> list[int]:
        try:
            return [
                int(idx)
                for idx in self.DynamicObject.get_ingredient_idx_list(
                    np.asarray(int(np.asarray(state.recipe)))
                )
            ]
        except Exception:
            return []

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
        pots = self._target_positions(
            static=static, dynamic=dynamic, target_kind="pot"
        )
        if not pots:
            return "none"
        x, y = min(pots, key=lambda pos: abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1]))
        pot_dynamic = int(dynamic[y, x])
        cooked = bool(pot_dynamic & int(self.DynamicObject.COOKED))
        timer = int(extra[y, x])
        count = self._ingredient_count(pot_dynamic)
        return f"nearest_pot ingredients={count} cooked={cooked} timer={timer}"

    def _pot_summaries(
        self, *, static: np.ndarray, dynamic: np.ndarray, extra: np.ndarray
    ) -> str:
        pots = self._target_positions(
            static=static, dynamic=dynamic, target_kind="pot"
        )
        if not pots:
            return "none"
        summaries = []
        for x, y in pots[:4]:
            pot_dynamic = int(dynamic[y, x])
            cooked = bool(pot_dynamic & int(self.DynamicObject.COOKED))
            timer = int(extra[y, x])
            ingredients = self._ingredient_names(pot_dynamic)
            ingredient_text = ",".join(ingredients) if ingredients else "empty"
            summaries.append(
                f"({x},{y}) ingredients={ingredient_text} cooked={cooked} timer={timer}"
            )
        return "; ".join(summaries)

    def _max_pot_ingredient_count(self, *, static: np.ndarray, dynamic: np.ndarray) -> int:
        pots = self._target_positions(
            static=static, dynamic=dynamic, target_kind="pot"
        )
        if not pots:
            return 0
        return max(self._ingredient_count(int(dynamic[y, x])) for x, y in pots)

    def _loose_object_summary(self, *, dynamic: np.ndarray) -> str:
        summaries = []
        plate_count = int(np.sum((dynamic & int(self.DynamicObject.PLATE)) > 0))
        cooked_count = int(np.sum((dynamic & int(self.DynamicObject.COOKED)) > 0))
        ingredient_cells = int(np.sum((dynamic & self._ingredient_mask()) > 0))
        if plate_count:
            summaries.append(f"plates={plate_count}")
        if cooked_count:
            summaries.append(f"cooked={cooked_count}")
        if ingredient_cells:
            summaries.append(f"ingredient_cells={ingredient_cells}")
        return ", ".join(summaries) if summaries else "none"

    def _nearby_static_summary(
        self, *, static: np.ndarray, state, agent_index: int
    ) -> str:
        x, y = self._agent_pos(state, agent_index)
        names = []
        for label, (dx, dy) in {
            "right": (1, 0),
            "down": (0, 1),
            "left": (-1, 0),
            "up": (0, -1),
        }.items():
            pos = (x + dx, y + dy)
            if (
                pos[0] < 0
                or pos[1] < 0
                or pos[1] >= static.shape[0]
                or pos[0] >= static.shape[1]
            ):
                names.append(f"{label}=bounds")
                continue
            names.append(f"{label}={self._static_name(int(static[pos[1], pos[0]]))}")
        return ", ".join(names)

    def _static_name(self, value: int) -> str:
        if value == int(self.StaticObject.EMPTY):
            return "empty"
        if value == int(self.StaticObject.WALL):
            return "wall"
        if value == int(self.StaticObject.POT):
            return "pot"
        if value == int(self.StaticObject.GOAL):
            return "goal"
        if value == int(self.StaticObject.PLATE_PILE):
            return "plate_pile"
        if value >= int(self.StaticObject.INGREDIENT_PILE_BASE):
            return f"ingredient_pile_{value - int(self.StaticObject.INGREDIENT_PILE_BASE)}"
        return f"static_{value}"

    def _ingredient_names(self, dynamic_value: int) -> list[str]:
        try:
            ingredients = self.DynamicObject.get_ingredient_idx_list(
                np.asarray(dynamic_value)
            )
        except Exception:
            ingredients = []
        return [f"ingredient_{idx}" for idx in ingredients]

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
