from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable

from nanocoop.schema import EvalEpisodeResult


def summarize_eval(results: Iterable[EvalEpisodeResult]) -> dict[str, float | int]:
    rows = list(results)
    if not rows:
        return {
            "primary_score": 0.0,
            "cross_play_mean_reward": 0.0,
            "self_play_mean_reward": 0.0,
            "mean_completion_rate": 0.0,
            "cross_partner_std": 0.0,
            "num_eval_episodes": 0,
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
    return {
        "primary_score": round(cross_play_mean, 4),
        "cross_play_mean_reward": round(cross_play_mean, 4),
        "self_play_mean_reward": round(self_play_mean, 4),
        "mean_completion_rate": round(completion_rate, 4),
        "cross_partner_std": round(pstdev(partner_means), 4) if len(partner_means) > 1 else 0.0,
        "num_eval_episodes": len(rows),
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
    lines.append("")
    return "\n".join(lines)
