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
    metrics["failed_episodes"] = [
        {
            "episode_id": row.episode_id,
            "layout": row.layout,
            "partner_name": row.partner_name,
            "seed": row.seed,
            "reward": row.total_reward,
            "step_count": row.step_count,
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


def run_contract_metadata(config: dict) -> dict:
    eval_cfg = config.get("eval", {})
    env_timeout = os.getenv("NANOCOOP_TIMEOUT_SECONDS")
    timeout_seconds = 180 if env_timeout is None else int(env_timeout)
    selected = selected_episode_ids(config)
    return {
        "benchmark_version": "v0.1",
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
