from __future__ import annotations

from typing import Any

from nanocoop.envs import make_backend
from nanocoop.partner_zoo import make_partner
from nanocoop.policy import HybridLookupPolicy, RemoteChatPolicy
from nanocoop.schema import EvalEpisodeResult, PolicyPackage


def package_to_policy(
    package: PolicyPackage,
    config: dict[str, Any],
    *,
    exploration_rate: float = 0.0,
    rng_seed: int = 0,
):
    model_cfg = config.get("model", {})
    api_base = model_cfg.get("api_base")
    if api_base:
        return RemoteChatPolicy.from_config(package, config)
    return HybridLookupPolicy(package=package, exploration_rate=exploration_rate, rng_seed=rng_seed)


def evaluate_package(
    package: PolicyPackage,
    config: dict[str, Any],
    *,
    include_self_play: bool = True,
) -> list[EvalEpisodeResult]:
    backend = make_backend(config)
    env_cfg = config.get("env", {})
    layouts = list(env_cfg.get("eval_layouts", []))
    seeds = list(env_cfg.get("eval_seeds", []))
    partner_names = list(config.get("partner_zoo", {}).get("eval", []))

    policy = package_to_policy(package, config)
    results: list[EvalEpisodeResult] = []

    for partner_name in partner_names:
        for seed in seeds:
            for layout in layouts:
                partner = make_partner(partner_name, seed=seed)
                trace = backend.rollout(
                    focal_policy=policy,
                    partner_policy=partner,
                    layout=layout,
                    seed=seed,
                    partner_name=partner_name,
                    mode="cross_play",
                )
                results.append(
                    EvalEpisodeResult(
                        layout=layout,
                        partner_name=partner_name,
                        seed=seed,
                        total_reward=trace.total_reward,
                        success=trace.success,
                        mode="cross_play",
                    )
                )

    if include_self_play:
        self_policy = package_to_policy(package, config, rng_seed=999)
        for seed in seeds:
            for layout in layouts:
                trace = backend.rollout(
                    focal_policy=self_policy,
                    partner_policy=self_policy,
                    layout=layout,
                    seed=seed,
                    partner_name="self",
                    mode="self_play",
                )
                results.append(
                    EvalEpisodeResult(
                        layout=layout,
                        partner_name="self",
                        seed=seed,
                        total_reward=trace.total_reward,
                        success=trace.success,
                        mode="self_play",
                    )
                )

    return results
