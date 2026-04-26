from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanocoop.schema import EpisodeTrace, Observation, StepRecord
from dungeongrid import DungeonGridEnvironment


class DungeonGridBackend:
    """NanoCoop adapter for the text-only DM + hero-party crawler."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        env_cfg = config.get("env", {})
        self.max_steps = int(env_cfg.get("max_steps", 120))
        self.num_heroes = int(env_cfg.get("num_heroes", 4))
        self.observation_mode = str(env_cfg.get("observation_mode", "mixed"))
        self.player_count_mode = str(
            env_cfg.get("player_count_mode") or self._player_count_mode(self.num_heroes)
        )
        self.warden_policy = self._make_warden_policy(config)

    def rollout(
        self,
        focal_policy,
        partner_policy,
        layout: str,
        seed: int,
        partner_name: str,
        mode: str,
        capture_states: bool = False,
    ) -> EpisodeTrace:
        env = DungeonGridEnvironment()
        obs = env.reset(
            quest_id=layout,
            num_heroes=self.num_heroes,
            seed=seed,
            observation_mode=self.observation_mode,
        )
        steps: list[StepRecord] = []
        total_reward = 0.0
        last_hero_action: str | None = None
        last_warden_action: str | None = None
        plan_records: list[dict[str, Any]] = []
        skipped_illegal_actions = 0
        executed_action_count = 0
        per_hero_action_counts: dict[str, int] = {}
        replay_frames: list[dict[str, Any]] = []
        if capture_states:
            replay_frames.append(
                self._replay_frame(
                    env=env,
                    step_index=0,
                    agent_id=obs.active_agent,
                    executed_actions=[],
                    skipped_actions=[],
                    unused_actions=[],
                    reward=0.0,
                    events=["initial state"],
                    new_achievements=[],
                )
            )

        for step_index in range(self.max_steps):
            active_agent = obs.active_agent
            legal = env._legal_actions(active_agent)
            if not legal:
                break
            policy_obs = self._to_nanocoop_observation(
                obs=obs,
                legal_actions=legal,
                layout=layout,
                step_index=step_index,
                last_hero_action=last_hero_action,
                last_warden_action=last_warden_action,
                partner_name=partner_name,
            )
            raw_plan = (
                [self._warden_action(env, partner_policy, policy_obs)]
                if active_agent == "warden"
                else self._policy_plan(focal_policy, policy_obs, active_agent)
            )
            plan = self._coerce_plan(raw_plan)
            executed: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []
            unused: list[dict[str, Any]] = []
            turn_reward = 0.0
            turn_events: list[str] = []
            pre_turn_agent = active_agent

            if active_agent == "warden":
                for plan_index, planned_action in enumerate(plan):
                    current_legal = env._legal_actions(pre_turn_agent)
                    if not current_legal or env.state.active_agent() != pre_turn_agent:
                        unused.extend(plan[plan_index:])
                        break
                    action = self._resolve_planned_action(planned_action, current_legal, pre_turn_agent)
                    if action is None:
                        skipped.append(planned_action)
                        continue
                    for field in (
                        "warden_policy",
                        "warden_intent",
                        "warden_axis_pressure",
                        "warden_fairness_check",
                        "warden_fallback_reason",
                    ):
                        if field in planned_action:
                            action[field] = planned_action[field]
                    step = env.step(action)
                    obs = step.observation
                    executed.append(action)
                    turn_reward += float(step.reward)
                    turn_events.append(str(step.info.get("narration") or step.info))
                    executed_action_count += 1
                    if step.done or env.state.active_agent() != pre_turn_agent:
                        unused.extend(plan[plan_index + 1 :])
                        break
            else:
                plan_result = env.act_plan(plan, intent=self._extract_plan_intent(raw_plan), agent_id=active_agent)
                obs = plan_result.observation
                executed = [
                    {"agent_id": active_agent, **action}
                    for action in plan_result.executed_actions
                ]
                skipped = list(plan_result.skipped_actions)
                unused = list(plan_result.unused_actions)
                turn_reward = float(plan_result.reward)
                turn_events.append(
                    json.dumps(
                        {
                            "reveal_stopped": plan_result.reveal_stopped,
                            "reveal_reason": plan_result.reveal_reason,
                            "skipped": skipped,
                            "new_achievements": plan_result.new_achievements,
                        },
                        sort_keys=True,
                    )
                )
                executed_action_count += len(executed)
                per_hero_action_counts[pre_turn_agent] = (
                    per_hero_action_counts.get(pre_turn_agent, 0) + len(executed)
                )

            if not executed:
                fallback_type = "warden_auto" if active_agent == "warden" else "end_turn"
                fallback = self._find_legal_action(
                    env._legal_actions(active_agent), {"type": fallback_type}
                )
                if fallback is None:
                    fallback = self._find_legal_action(
                        env._legal_actions(active_agent), {"type": "end_turn"}
                    )
                if fallback is not None and env.state.active_agent() == active_agent:
                    step = env.step({"agent_id": active_agent, **fallback})
                    obs = step.observation
                    executed.append({"agent_id": active_agent, **fallback})
                    turn_reward += float(step.reward)
                    turn_events.append(str(step.info.get("narration") or step.info))

            total_reward += turn_reward
            skipped_illegal_actions += len(skipped)
            if active_agent == "warden":
                last_warden_action = self._compact_actions(executed)
            else:
                last_hero_action = f"{active_agent}:{self._compact_actions(executed)}"
            plan_record = {
                "step_index": step_index,
                "agent_id": active_agent,
                "role": policy_obs.metadata.get("role"),
                "submitted_plan": plan,
                "executed_actions": executed,
                "skipped_illegal_actions": skipped,
                "unused_actions": unused,
                "reward": round(turn_reward, 4),
                "events": turn_events,
                "new_achievements": self._new_achievements_from_events(turn_events),
            }
            if active_agent == "warden":
                source = executed[0] if executed else (plan[0] if plan else {})
                if isinstance(source, dict):
                    plan_record.update(
                        {
                            "warden_policy": source.get("warden_policy"),
                            "warden_intent": source.get("warden_intent"),
                            "warden_axis_pressure": source.get("warden_axis_pressure"),
                            "warden_fairness_check": source.get("warden_fairness_check"),
                            "warden_fallback_reason": source.get("warden_fallback_reason"),
                        }
                    )
            plan_records.append(plan_record)
            if capture_states:
                replay_frames.append(
                    self._replay_frame(
                        env=env,
                        step_index=step_index,
                        agent_id=pre_turn_agent,
                        executed_actions=executed,
                        skipped_actions=skipped,
                        unused_actions=unused,
                        reward=round(turn_reward, 4),
                        events=turn_events,
                        new_achievements=plan_record["new_achievements"],
                    )
                )
            steps.append(
                StepRecord(
                    step_index=step_index,
                    action_agent_0=self._compact_actions(executed)
                    if active_agent != "warden"
                    else last_hero_action or "none",
                    action_agent_1=self._compact_actions(executed)
                    if active_agent == "warden"
                    else last_warden_action or "none",
                    reward=round(turn_reward, 4),
                    event=json.dumps(plan_record, sort_keys=True),
                    focal_observation=policy_obs,
                )
            )
            if env.state.done:
                break

        if capture_states:
            final_state = env.public_state_json()
            if not replay_frames or replay_frames[-1].get("state", {}).get("trace_len") != final_state.get("trace_len"):
                replay_frames.append(
                    {
                        "state": final_state,
                        "step_index": len(steps),
                        "agent_id": env.state.active_agent(),
                        "executed_actions": [],
                        "skipped_actions": [],
                        "unused_actions": [],
                        "reward": 0.0,
                        "events": ["final state"],
                        "new_achievements": [],
                    }
                )

        metrics = env.agent_engine.metrics(env.state)
        transcript = env.export_transcript()
        final_scout_reward = self._final_scout_reward(metrics)
        total_reward += final_scout_reward
        metrics["final_scout_reward"] = round(final_scout_reward, 4)
        llm_calls = int(getattr(focal_policy, "llm_call_count", 0) or 0)
        llm_usage = dict(getattr(focal_policy, "llm_usage", {}) or {})
        llm_usage_events = list(getattr(focal_policy, "llm_usage_events", []) or [])
        private_plan_tool_count = int(getattr(focal_policy, "private_plan_tool_count", 0) or 0)
        private_plan_tool_counts = dict(getattr(focal_policy, "private_plan_tool_counts", {}) or {})
        private_plans = dict(getattr(focal_policy, "_private_plans", {}) or {})
        warden_policy = self.warden_policy
        warden_llm_calls = int(getattr(warden_policy, "llm_call_count", 0) or 0)
        warden_llm_usage = dict(getattr(warden_policy, "llm_usage", {}) or {})
        warden_llm_usage_events = list(getattr(warden_policy, "llm_usage_events", []) or [])
        warden_fallback_count = int(getattr(warden_policy, "fallback_count", 0) or 0)
        warden_action_counts = dict(getattr(warden_policy, "action_counts", {}) or {})
        warden_decisions = list(getattr(warden_policy, "decisions", []) or [])
        metrics.update(
            {
                "skipped_illegal_actions": skipped_illegal_actions,
                "executed_action_count": executed_action_count,
                "per_hero_action_counts": per_hero_action_counts,
                "player_count": self.num_heroes,
                "player_count_mode": self.player_count_mode,
                "warden_policy_kind": self._warden_policy_kind(),
                "warden_llm_call_count": warden_llm_calls,
                "warden_llm_usage": warden_llm_usage,
                "warden_fallback_count": warden_fallback_count,
                "warden_action_counts": warden_action_counts,
            }
        )
        return EpisodeTrace(
            layout=layout,
            seed=seed,
            partner_name=partner_name,
            total_reward=round(total_reward, 4),
            success=bool(metrics.get("success", False)),
            steps=steps,
            metadata={
                "backend": "dungeongrid",
                "mode": mode,
                "focal_llm_call_count": llm_calls,
                "focal_llm_usage": llm_usage,
                "focal_llm_usage_events": llm_usage_events,
                "warden_policy_kind": self._warden_policy_kind(),
                "warden_llm_call_count": warden_llm_calls,
                "warden_llm_usage": warden_llm_usage,
                "warden_llm_usage_events": warden_llm_usage_events,
                "warden_fallback_count": warden_fallback_count,
                "warden_action_counts": warden_action_counts,
                "warden_decisions": warden_decisions,
                "private_plan_tool_count": private_plan_tool_count,
                "private_plan_tool_counts": private_plan_tool_counts,
                "private_plans": private_plans,
                "dungeongrid_metrics": metrics,
                "dungeongrid_transcript": transcript,
                "player_count": self.num_heroes,
                "player_count_mode": self.player_count_mode,
                "plan_records": plan_records,
                "capture_states": capture_states,
                "replay_frames": replay_frames,
            },
        )

    def _make_warden_policy(self, config: dict[str, Any]):
        warden_cfg = config.get("warden_policy")
        deterministic_kinds = {
            "deterministic",
            "deterministic_partner",
            "deterministic_warden",
            "scripted",
            "scripted_warden",
            "none",
            "off",
            "false",
        }
        if warden_cfg is False:
            return None
        if isinstance(warden_cfg, str) and warden_cfg.lower() in deterministic_kinds:
            return None
        if isinstance(warden_cfg, dict):
            kind = str(warden_cfg.get("kind") or "dungeongrid_warden_react").lower()
            if kind in deterministic_kinds:
                return None
            if kind != "dungeongrid_warden_react":
                return None
            policy_config = config
        elif warden_cfg is None:
            policy_cfg = config.get("policy", {}) if isinstance(config.get("policy"), dict) else {}
            policy_kind = str(policy_cfg.get("kind") or "").lower()
            if policy_kind not in {"dungeongrid_react", "torchgrid_react"}:
                return None
            policy_config = {
                **config,
                "warden_policy": {
                    "kind": "dungeongrid_warden_react",
                    "fallback": "deterministic_warden",
                },
            }
        else:
            return None
        from nanocoop.policy import DungeonGridWardenReActPolicy

        return DungeonGridWardenReActPolicy.from_config(policy_config)

    def _warden_policy_kind(self) -> str:
        if self.warden_policy is not None:
            return "dungeongrid_warden_react"
        return "deterministic_partner"

    def _warden_action(self, env: DungeonGridEnvironment, partner_policy, policy_obs: Observation):
        if self.warden_policy is not None:
            return self.warden_policy.act(env.observe_warden())
        return partner_policy.act(policy_obs)

    def write_rollout_gif(self, trace: EpisodeTrace, path: Path) -> Path:
        self._write_rollout_text(trace, path.with_suffix(".txt"))
        replay_frames = trace.metadata.get("replay_frames")
        if isinstance(replay_frames, list) and replay_frames:
            from dungeongrid.rendering import (
                render_sprite_gif,
                render_sprite_html,
                render_terminal_gif,
                render_terminal_html,
            )

            gif_path = Path(path)
            render_terminal_gif(replay_frames, gif_path)
            render_terminal_html(
                replay_frames,
                gif_path.with_suffix(".html"),
                title=f"DungeonGrid Replay: {trace.layout} seed={trace.seed}",
            )
            sprite_path = gif_path.with_name(f"{gif_path.stem}_sprite.gif")
            render_sprite_gif(replay_frames, sprite_path)
            render_sprite_html(
                replay_frames,
                sprite_path.with_suffix(".html"),
                title=f"DungeonGrid Sprite Replay: {trace.layout} seed={trace.seed}",
            )
            return gif_path
        return path.with_suffix(".txt")

    def _write_rollout_text(self, trace: EpisodeTrace, text_path: Path) -> Path:
        text_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{trace.layout} seed={trace.seed} partner={trace.partner_name}"]
        transcript = trace.metadata.get("dungeongrid_transcript")
        if isinstance(transcript, dict):
            metrics = transcript.get("metrics", {})
            lines.append(
                "summary: "
                f"success={metrics.get('success')} "
                f"achievements={metrics.get('achievement_count')} "
                f"reward={trace.total_reward}"
            )
            for event in transcript.get("achievements", []):
                lines.append(f"achievement: {event.get('id')} {event.get('title')} +{event.get('reward')}")
            lines.append("")
        lines.extend(f"{row.step_index}: {row.event}" for row in trace.steps)
        text_path.write_text("\n".join(lines), encoding="utf-8")
        return text_path

    def _replay_frame(
        self,
        *,
        env: DungeonGridEnvironment,
        step_index: int,
        agent_id: str,
        executed_actions: list[dict[str, Any]],
        skipped_actions: list[dict[str, Any]],
        unused_actions: list[dict[str, Any]],
        reward: float,
        events: list[str],
        new_achievements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "state": env.public_state_json(),
            "step_index": step_index,
            "agent_id": agent_id,
            "executed_actions": list(executed_actions),
            "skipped_actions": list(skipped_actions),
            "unused_actions": list(unused_actions),
            "reward": reward,
            "events": list(events),
            "new_achievements": list(new_achievements),
        }

    def _new_achievements_from_events(self, events: list[str]) -> list[dict[str, Any]]:
        achievements: list[dict[str, Any]] = []
        for event in events:
            try:
                parsed = json.loads(event)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                achievements.extend(parsed.get("new_achievements", []) or [])
        return achievements

    def _to_nanocoop_observation(
        self,
        *,
        obs,
        legal_actions: list[dict[str, Any]],
        layout: str,
        step_index: int,
        last_hero_action: str | None,
        last_warden_action: str | None,
        partner_name: str,
    ) -> Observation:
        legal_labels = tuple(self._action_label(action) for action in legal_actions)
        symbolic = dict(obs.symbolic)
        symbolic.update(
            {
                "backend": "dungeongrid",
                "text": obs.text,
                "visible_map": obs.visible_map,
                "active_agent": obs.active_agent,
                "ap_remaining": obs.symbolic.get("ap_remaining"),
                "internal_action_objects": list(legal_actions),
                "role": (symbolic.get("self") or {}).get("role")
                if isinstance(symbolic.get("self"), dict)
                else None,
                "partner_name": partner_name,
            }
        )
        return Observation(
            agent_id=obs.agent_id,
            layout=layout,
            step_index=step_index,
            max_steps=self.max_steps,
            private_recipe=None,
            private_pot=None,
            shared_recipe_known=True,
            shared_pot_known=True,
            ingredient_ready=False,
            pot_ready=False,
            dish_ready=False,
            plated=False,
            delivered=False,
            last_partner_action=last_warden_action,
            last_joint_event=last_hero_action,
            available_actions=legal_labels,
            convention_hint=(
                "Coordinate the Barbarian, Wizard, Elf, and Dwarf as a party; "
                "the Warden controls dungeon pressure."
            ),
            recent_events=tuple(obs.symbolic.get("event_log", [])[-3:])
            if isinstance(obs.symbolic.get("event_log"), list)
            else (),
            metadata=symbolic,
        )

    def _extract_plan_intent(self, raw_plan: Any) -> str | None:
        if isinstance(raw_plan, dict) and isinstance(raw_plan.get("intent"), str):
            return str(raw_plan["intent"])
        if isinstance(raw_plan, list) and raw_plan and isinstance(raw_plan[0], dict):
            intent = raw_plan[0].get("intent")
            if isinstance(intent, str):
                return intent
        return None

    def _policy_plan(self, policy, observation: Observation, active_agent: str) -> list[Any]:
        policy = self._policy_for_active_hero(policy, observation, active_agent)
        act_plan = getattr(policy, "act_plan", None)
        if callable(act_plan):
            return list(act_plan(observation))
        return [policy.act(observation)]

    def _policy_for_active_hero(self, policy, observation: Observation, active_agent: str):
        role = str(observation.metadata.get("role") or "")
        if isinstance(policy, dict):
            if not policy:
                raise ValueError("DungeonGrid focal policy mapping cannot be empty")
            return (
                policy.get(active_agent)
                or policy.get(role)
                or policy.get("default")
                or next(iter(policy.values()))
            )
        policy_for_agent = getattr(policy, "policy_for_agent", None)
        if callable(policy_for_agent):
            return policy_for_agent(active_agent=active_agent, role=role, observation=observation)
        return policy

    def _coerce_plan(self, raw_plan: Any) -> list[dict[str, Any]]:
        if isinstance(raw_plan, dict):
            if isinstance(raw_plan.get("actions"), list):
                raw_plan = raw_plan["actions"]
            else:
                raw_plan = [raw_plan]
        if isinstance(raw_plan, str):
            parsed = self._parse_action(raw_plan)
            if isinstance(parsed, dict) and isinstance(parsed.get("actions"), list):
                raw_plan = parsed["actions"]
            elif parsed is not None:
                raw_plan = [parsed]
            else:
                raw_plan = []
        if not isinstance(raw_plan, list):
            return []
        result: list[dict[str, Any]] = []
        for item in raw_plan:
            if isinstance(item, dict) and item.get("type"):
                result.append(dict(item))
            elif isinstance(item, str):
                parsed = self._parse_action(item)
                if parsed is not None and parsed.get("type"):
                    result.append(parsed)
        return result

    def _parse_action(self, raw_action: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_action)
        except json.JSONDecodeError:
            return {"type": raw_action} if raw_action else None
        if isinstance(payload, dict):
            return payload
        return None

    def _resolve_planned_action(
        self, planned_action: dict[str, Any], legal: list[dict[str, Any]], active_agent: str
    ) -> dict[str, Any] | None:
        match = self._find_legal_action(legal, planned_action)
        if match is None:
            return None
        resolved = dict(match)
        if (
            resolved.get("type") == "message"
            and isinstance(planned_action.get("payload"), dict)
        ):
            resolved["payload"] = dict(planned_action["payload"])
        return {"agent_id": active_agent, **resolved}

    def _find_legal_action(
        self, legal: list[dict[str, Any]], requested: dict[str, Any]
    ) -> dict[str, Any] | None:
        for action in legal:
            if self._action_matches(requested, action):
                return action
        return None

    def _action_matches(self, requested: dict[str, Any], legal: dict[str, Any]) -> bool:
        if requested.get("type") != legal.get("type"):
            return False
        for key in ("direction", "target", "payload"):
            if key in legal and key not in requested:
                return False
            if key in legal and requested.get(key) != legal.get(key):
                return False
        return True

    def _action_label(self, action: dict[str, Any]) -> str:
        return json.dumps(action, sort_keys=True, separators=(",", ":"))

    def _compact_actions(self, actions: list[dict[str, Any]]) -> str:
        if not actions:
            return "none"
        return json.dumps(actions, sort_keys=True, separators=(",", ":"))

    def _player_count_mode(self, num_heroes: int) -> str:
        return {1: "solo", 2: "duo", 3: "trio", 4: "squad"}.get(num_heroes, str(num_heroes))

    def _final_scout_reward(self, metrics: dict[str, Any]) -> float:
        env_cfg = self.config.get("env", {})
        room_reward = float(env_cfg.get("room_exploration_reward", 1.0))
        tile_coverage_reward = float(env_cfg.get("tile_coverage_reward", 0.0))
        rooms_explored = float(metrics.get("rooms_explored", 0.0) or 0.0)
        exploration = float(metrics.get("exploration", 0.0) or 0.0)
        return rooms_explored * room_reward + exploration * tile_coverage_reward
