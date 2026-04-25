from __future__ import annotations

import os
from statistics import mean, pstdev
from typing import Iterable, Sequence

from nanocoop.episode_plan import build_cross_play_episodes, selected_episode_ids
from nanocoop.schema import EvalEpisodeResult


def summarize_eval(results: Iterable[EvalEpisodeResult]) -> dict:
    rows = list(results)
    if not rows:
        return {
            "primary_score": 0.0,
            "cross_play_mean_reward": 0.0,
            "self_play_mean_reward": 0.0,
            "mean_completion_rate": 0.0,
            "cross_partner_std": 0.0,
            "num_eval_episodes": 0,
            "layout_breakdown": {},
            "partner_breakdown": {},
            "failed_episodes": [],
        }

    cross = [row.total_reward for row in rows if row.mode == "cross_play"]
    self_play = [row.total_reward for row in rows if row.mode == "self_play"]
    partner_rewards: dict[str, list[float]] = {}
    for row in rows:
        if row.mode != "cross_play":
            continue
        partner_rewards.setdefault(row.partner_name, []).append(row.total_reward)

    partner_means = [mean(values) for values in partner_rewards.values()] or [0.0]
    cross_play_mean = mean(cross) if cross else 0.0
    self_play_mean = mean(self_play) if self_play else 0.0
    completion_rate = mean(1.0 if row.success else 0.0 for row in rows)
    metrics = {
        "primary_score": round(cross_play_mean, 4),
        "cross_play_mean_reward": round(cross_play_mean, 4),
        "self_play_mean_reward": round(self_play_mean, 4),
        "mean_completion_rate": round(completion_rate, 4),
        "cross_partner_std": round(pstdev(partner_means), 4) if len(partner_means) > 1 else 0.0,
        "num_eval_episodes": len(rows),
    }
    metrics["layout_breakdown"] = _breakdown(rows, key="layout")
    metrics["partner_breakdown"] = _breakdown(
        [row for row in rows if row.mode == "cross_play"], key="partner_name"
    )
    player_count_rows = [
        row for row in rows if row.metadata.get("player_count_mode") is not None
    ]
    if player_count_rows:
        cross_completion = (
            mean(1.0 if row.success else 0.0 for row in rows if row.mode == "cross_play")
            if cross
            else 0.0
        )
        metrics["primary_score"] = round(cross_completion * 100.0 + cross_play_mean, 4)
        metrics["objective_first_completion_component"] = round(cross_completion, 4)
        metrics["player_count_breakdown"] = _metadata_breakdown(
            player_count_rows, metadata_key="player_count_mode"
        )
        metrics["dungeongrid_secondary_metrics"] = _dungeongrid_secondary_metrics(rows)
    metrics["failed_episodes"] = [
        {
            "episode_id": row.episode_id,
            "layout": row.layout,
            "partner_name": row.partner_name,
            "seed": row.seed,
            "reward": row.total_reward,
            "step_count": row.step_count,
            "player_count_mode": row.metadata.get("player_count_mode"),
        }
        for row in rows
        if not row.success
    ]
    return metrics


def _breakdown(rows: Sequence[EvalEpisodeResult], *, key: str) -> dict[str, dict]:
    groups: dict[str, list[EvalEpisodeResult]] = {}
    for row in rows:
        groups.setdefault(str(getattr(row, key)), []).append(row)
    return {
        group_key: {
            "mean_reward": round(mean(row.total_reward for row in group_rows), 4),
            "completion_rate": round(
                mean(1.0 if row.success else 0.0 for row in group_rows), 4
            ),
            "num_episodes": len(group_rows),
        }
        for group_key, group_rows in sorted(groups.items())
    }


def _metadata_breakdown(rows: Sequence[EvalEpisodeResult], *, metadata_key: str) -> dict[str, dict]:
    groups: dict[str, list[EvalEpisodeResult]] = {}
    for row in rows:
        groups.setdefault(str(row.metadata.get(metadata_key)), []).append(row)
    return {
        group_key: {
            "mean_reward": round(mean(row.total_reward for row in group_rows), 4),
            "completion_rate": round(
                mean(1.0 if row.success else 0.0 for row in group_rows), 4
            ),
            "num_episodes": len(group_rows),
        }
        for group_key, group_rows in sorted(groups.items())
    }


def _dungeongrid_secondary_metrics(rows: Sequence[EvalEpisodeResult]) -> dict:
    metric_rows = [
        row.metadata.get("dungeongrid_metrics", {})
        for row in rows
        if row.metadata.get("dungeongrid_metrics")
    ]
    if not metric_rows:
        return {}

    def avg(key: str) -> float:
        values = [float(metrics.get(key, 0.0) or 0.0) for metrics in metric_rows]
        return round(mean(values), 4) if values else 0.0

    per_hero_counts: dict[str, list[float]] = {}
    per_hero_rewards: dict[str, list[float]] = {}
    per_hero_invalids: dict[str, list[float]] = {}
    per_hero_achievements: dict[str, list[float]] = {}
    achievement_counts: dict[str, int] = {}
    for metrics in metric_rows:
        for hero, count in dict(metrics.get("per_hero_action_counts", {})).items():
            per_hero_counts.setdefault(str(hero), []).append(float(count))
        for hero, stats in dict(metrics.get("per_hero_stats", {})).items():
            if not isinstance(stats, dict):
                continue
            hero_id = str(hero)
            per_hero_rewards.setdefault(hero_id, []).append(float(stats.get("reward", 0.0) or 0.0))
            per_hero_invalids.setdefault(hero_id, []).append(
                float(stats.get("invalid_actions", 0.0) or 0.0)
            )
            per_hero_achievements.setdefault(hero_id, []).append(
                float(
                    stats.get(
                        "achievement_count",
                        len(stats.get("achievements_unlocked", []) or []),
                    )
                    or 0.0
                )
            )
        for achievement_id in metrics.get("achievements_unlocked", []) or []:
            achievement_counts[str(achievement_id)] = achievement_counts.get(str(achievement_id), 0) + 1

    return {
        "mean_survival": avg("survival"),
        "mean_rounds": avg("rounds"),
        "mean_exploration": avg("exploration"),
        "mean_room_exploration": avg("room_exploration"),
        "mean_rooms_explored": avg("rooms_explored"),
        "mean_scout_reward": avg("scout_reward"),
        "mean_final_scout_reward": avg("final_scout_reward"),
        "mean_treasure": avg("treasure"),
        "mean_monsters_defeated": avg("monsters_defeated"),
        "mean_invalid_actions": avg("invalid_actions"),
        "mean_skipped_illegal_actions": avg("skipped_illegal_actions"),
        "mean_executed_action_count": avg("executed_action_count"),
        "mean_achievement_count": avg("achievement_count"),
        "mean_quest_achievement_count": avg("quest_achievement_count"),
        "mean_global_achievement_count": avg("global_achievement_count"),
        "mean_achievement_reward": avg("achievement_reward"),
        "achievement_frequencies": {
            achievement_id: round(count / max(1, len(metric_rows)), 4)
            for achievement_id, count in sorted(achievement_counts.items())
        },
        "mean_per_hero_action_counts": {
            hero: round(mean(values), 4) for hero, values in sorted(per_hero_counts.items())
        },
        "mean_per_hero_reward": {
            hero: round(mean(values), 4) for hero, values in sorted(per_hero_rewards.items())
        },
        "mean_per_hero_invalid_actions": {
            hero: round(mean(values), 4) for hero, values in sorted(per_hero_invalids.items())
        },
        "mean_per_hero_achievements": {
            hero: round(mean(values), 4)
            for hero, values in sorted(per_hero_achievements.items())
        },
    }


def run_contract_metadata(config: dict) -> dict:
    eval_cfg = config.get("eval", {})
    env_timeout = os.getenv("NANOCOOP_TIMEOUT_SECONDS")
    timeout_seconds = 180 if env_timeout is None else int(env_timeout)
    selected = selected_episode_ids(config)
    return {
        "benchmark_version": "v0.1",
        "player_count_mode": config.get("env", {}).get("player_count_mode"),
        "player_count": config.get("env", {}).get("num_heroes"),
        "official_episode_count": int(eval_cfg.get("default_episode_count", len(selected))),
        "official_episode_ids": selected,
        "expanded_episode_count": len(build_cross_play_episodes(config)),
        "timeout_seconds": timeout_seconds,
        "timed_out": False,
        "timeout_mode": "official_disabled" if timeout_seconds == 0 else "dev_guard",
        "official_record": timeout_seconds == 0,
    }


def render_summary_markdown(
    *,
    track: str,
    run_name: str,
    metrics: dict,
    notes: list[str] | None = None,
) -> str:
    lines = [
        f"# {run_name}",
        "",
        f"- track: `{track}`",
        f"- primary score: `{metrics['primary_score']}`",
        f"- cross-play mean reward: `{metrics['cross_play_mean_reward']}`",
        f"- self-play mean reward: `{metrics['self_play_mean_reward']}`",
        f"- mean completion rate: `{metrics['mean_completion_rate']}`",
        f"- cross-partner std: `{metrics['cross_partner_std']}`",
        f"- num eval episodes: `{metrics['num_eval_episodes']}`",
    ]
    if notes:
        lines.append("")
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")
    layout_breakdown = metrics.get("layout_breakdown", {})
    if layout_breakdown:
        lines.append("")
        lines.append("## Layout Breakdown")
        for layout, values in layout_breakdown.items():
            lines.append(
                "- "
                f"`{layout}`: mean_reward=`{values['mean_reward']}`, "
                f"completion=`{values['completion_rate']}`, "
                f"episodes=`{values['num_episodes']}`"
            )
    partner_breakdown = metrics.get("partner_breakdown", {})
    if partner_breakdown:
        lines.append("")
        lines.append("## Partner Breakdown")
        for partner, values in partner_breakdown.items():
            lines.append(
                "- "
                f"`{partner}`: mean_reward=`{values['mean_reward']}`, "
                f"completion=`{values['completion_rate']}`, "
                f"episodes=`{values['num_episodes']}`"
            )
    player_count_breakdown = metrics.get("player_count_breakdown", {})
    if player_count_breakdown:
        lines.append("")
        lines.append("## Player Count Breakdown")
        for mode, values in player_count_breakdown.items():
            lines.append(
                "- "
                f"`{mode}`: mean_reward=`{values['mean_reward']}`, "
                f"completion=`{values['completion_rate']}`, "
                f"episodes=`{values['num_episodes']}`"
            )
    secondary = metrics.get("dungeongrid_secondary_metrics", {})
    if secondary:
        lines.append("")
        lines.append("## DungeonGrid Secondary Metrics")
        for key, value in secondary.items():
            lines.append(f"- `{key}`: `{value}`")
    failed_episodes = metrics.get("failed_episodes", [])
    if failed_episodes:
        lines.append("")
        lines.append("## Failed Episodes")
        for row in failed_episodes:
            lines.append(
                "- "
                f"episode `{row.get('episode_id')}`: "
                f"layout=`{row['layout']}`, partner=`{row['partner_name']}`, "
                f"seed=`{row['seed']}`, reward=`{row['reward']}`, "
                f"steps=`{row.get('step_count')}`"
            )
    lines.append("")
    return "\n".join(lines)
