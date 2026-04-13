from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalEpisode:
    episode_id: int
    layout: str
    partner_name: str
    seed: int


def build_cross_play_episodes(config: dict[str, Any]) -> list[EvalEpisode]:
    env_cfg = config.get("env", {})
    eval_cfg = config.get("eval", {})
    layouts = list(env_cfg.get("eval_layouts", []))
    partner_names = list(config.get("partner_zoo", {}).get("eval", []))
    seed_start = int(eval_cfg.get("episode_seed_start", 1))
    repeats = int(eval_cfg.get("episodes_per_combo", 4))

    episodes = []
    episode_id = 1
    for partner_name in partner_names:
        for layout in layouts:
            for _ in range(repeats):
                episodes.append(
                    EvalEpisode(
                        episode_id=episode_id,
                        layout=str(layout),
                        partner_name=str(partner_name),
                        seed=seed_start + episode_id - 1,
                    )
                )
                episode_id += 1
    return episodes


def selected_episode_ids(config: dict[str, Any]) -> list[int]:
    eval_cfg = config.get("eval", {})
    explicit_ids = eval_cfg.get("episode_ids")
    if explicit_ids:
        return [int(episode_id) for episode_id in explicit_ids]

    count = int(eval_cfg.get("default_episode_count", 20))
    seed = int(eval_cfg.get("default_episode_sample_seed", 20260412))
    episodes = build_cross_play_episodes(config)
    episode_ids = [episode.episode_id for episode in episodes]
    if count >= len(episode_ids):
        return episode_ids
    rng = random.Random(seed)
    return sorted(rng.sample(episode_ids, count))


def resolve_episode_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def select_cross_play_episodes(
    config: dict[str, Any], episode_ids: list[int] | None = None
) -> list[EvalEpisode]:
    episodes = build_cross_play_episodes(config)
    selected = set(episode_ids or selected_episode_ids(config))
    return [episode for episode in episodes if episode.episode_id in selected]
