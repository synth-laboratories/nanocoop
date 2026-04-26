from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from typing import Any

from nanocoop.envs import make_backend
from nanocoop.episode_plan import select_cross_play_episodes
from nanocoop.partner_zoo import make_partner
from nanocoop.policy import HybridLookupPolicy, RemoteChatPolicy, DungeonGridReActPolicy
from nanocoop.schema import EpisodeTrace, EvalEpisodeResult, PolicyPackage


def package_to_policy(
    package: PolicyPackage,
    config: dict[str, Any],
    *,
    exploration_rate: float = 0.0,
    rng_seed: int = 0,
):
    model_cfg = config.get("model", {})
    policy_cfg = config.get("policy", {})
    policy_kind = str(policy_cfg.get("kind", "")).lower()
    if policy_kind in {"dungeongrid_react", "react_dungeongrid"}:
        return DungeonGridReActPolicy.from_config(package, config)
    api_base = model_cfg.get("api_base")
    if api_base:
        return RemoteChatPolicy.from_config(package, config)
    return HybridLookupPolicy(package=package, exploration_rate=exploration_rate, rng_seed=rng_seed)


def evaluate_package(
    package: PolicyPackage,
    config: dict[str, Any],
    *,
    include_self_play: bool = True,
    episode_ids: list[int] | None = None,
    workers: int | None = None,
    progress: bool = False,
    rollout_trace_sink: dict[int, EpisodeTrace] | None = None,
) -> list[EvalEpisodeResult]:
    env_cfg = config.get("env", {})
    layouts = list(env_cfg.get("eval_layouts", []))
    eval_cfg = config.get("eval", {})
    selected_episodes = select_cross_play_episodes(config, episode_ids=episode_ids)
    worker_count = int(workers or eval_cfg.get("workers", 1))

    def run_cross_play(episode) -> EvalEpisodeResult:
        backend = make_backend(config)
        policy = package_to_policy(package, config, rng_seed=episode.episode_id)
        partner = make_partner(episode.partner_name, seed=episode.seed)
        trace = backend.rollout(
            focal_policy=policy,
            partner_policy=partner,
            layout=episode.layout,
            seed=episode.seed,
            partner_name=episode.partner_name,
            mode="cross_play",
            capture_states=rollout_trace_sink is not None,
        )
        if rollout_trace_sink is not None:
            rollout_trace_sink[episode.episode_id] = trace
        return EvalEpisodeResult(
            layout=episode.layout,
            partner_name=episode.partner_name,
            seed=episode.seed,
            total_reward=trace.total_reward,
            success=trace.success,
            mode="cross_play",
            episode_id=episode.episode_id,
            step_count=len(trace.steps),
            llm_call_count=int(trace.metadata.get("focal_llm_call_count", 0) or 0),
            metadata=trace.metadata,
        )

    if worker_count > 1 and len(selected_episodes) > 1:
        results_by_id = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_cross_play, episode): episode
                for episode in selected_episodes
            }
            for future in as_completed(futures):
                episode = futures[future]
                result = future.result()
                results_by_id[episode.episode_id] = result
                if progress:
                    done = len(results_by_id)
                    print(
                        (
                            f"episode {episode.episode_id} "
                            f"({done}/{len(selected_episodes)}): "
                            f"layout={episode.layout} partner={episode.partner_name} "
                            f"seed={episode.seed} reward={result.total_reward} "
                            f"success={result.success}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
        results = [results_by_id[episode.episode_id] for episode in selected_episodes]
    else:
        results = []
        for episode in selected_episodes:
            result = run_cross_play(episode)
            results.append(result)
            if progress:
                print(
                    (
                        f"episode {episode.episode_id} "
                        f"({len(results)}/{len(selected_episodes)}): "
                        f"layout={episode.layout} partner={episode.partner_name} "
                        f"seed={episode.seed} reward={result.total_reward} "
                        f"success={result.success}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    if include_self_play:
        backend = make_backend(config)
        self_policy = package_to_policy(package, config, rng_seed=999)
        self_seeds = list(env_cfg.get("self_play_seeds", [])) or [
            1000 + index for index, _ in enumerate(layouts, start=1)
        ]
        for seed in self_seeds:
            for layout in layouts:
                self_partner = self_policy
                self_partner_name = "self"
                if str(config.get("backend", "")).lower() == "dungeongrid":
                    self_partner_name = str(env_cfg.get("self_play_warden", "scripted_warden"))
                    self_partner = make_partner(self_partner_name, seed=seed)

                trace = backend.rollout(
                    focal_policy=self_policy,
                    partner_policy=self_partner,
                    layout=layout,
                    seed=seed,
                    partner_name=self_partner_name,
                    mode="self_play",
                )
                results.append(
                    EvalEpisodeResult(
                        layout=layout,
                        partner_name=self_partner_name,
                        seed=seed,
                        total_reward=trace.total_reward,
                        success=trace.success,
                        mode="self_play",
                        episode_id=None,
                        step_count=len(trace.steps),
                        llm_call_count=int(trace.metadata.get("focal_llm_call_count", 0) or 0),
                        metadata=trace.metadata,
                    )
                )

    return results
