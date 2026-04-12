from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from nanocoop.envs import make_backend
from nanocoop.evaluation import evaluate_package
from nanocoop.partner_zoo import make_partner
from nanocoop.policy import make_seed_package
from nanocoop.record_bundle import write_record_bundle
from nanocoop.schema import PolicyPackage
from nanocoop.score import render_summary_markdown, summarize_eval

# Local import helper for static analyzers.
try:
    from nanocoop.evaluation import package_to_policy as _package_to_policy
except Exception:  # pragma: no cover
    _package_to_policy = None


def _policy_from_package(package: PolicyPackage, config: dict[str, Any], exploration_rate: float, rng_seed: int):
    if _package_to_policy is None:
        from nanocoop.evaluation import package_to_policy as runtime_policy_from_package
        return runtime_policy_from_package(package, config, exploration_rate=exploration_rate, rng_seed=rng_seed)
    return _package_to_policy(package, config, exploration_rate=exploration_rate, rng_seed=rng_seed)


def run(config: dict[str, Any]) -> dict[str, Any]:
    backend = make_backend(config)
    env_cfg = config.get("env", {})
    rlvr_cfg = config.get("rlvr", {})
    track = str(config.get("track"))
    run_name = str(config.get("run_name"))

    package = make_seed_package(
        name=run_name,
        backend=str(config.get("backend", "mock")),
        prompt=str(config.get("policy", {}).get("seed_prompt", "")),
        model_name=str(config.get("model", {}).get("name", "")) or None,
    )

    train_layouts = list(env_cfg.get("train_layouts", []))
    train_seeds = list(env_cfg.get("train_seeds", []))
    partner_names = list(config.get("partner_zoo", {}).get("train", []))
    exploration_rate = float(rlvr_cfg.get("exploration_rate", 0.15))
    iterations = int(rlvr_cfg.get("iterations", 4))
    episodes_per_iteration = int(rlvr_cfg.get("episodes_per_iteration", 12))

    best_package = package
    best_score = float("-inf")
    iteration_notes: list[str] = []

    for iteration in range(iterations):
        policy = _policy_from_package(package, config, exploration_rate=exploration_rate, rng_seed=iteration)
        action_values: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        train_returns: list[float] = []

        for episode_index in range(episodes_per_iteration):
            layout = train_layouts[episode_index % len(train_layouts)]
            seed = train_seeds[episode_index % len(train_seeds)] + iteration * 100 + episode_index
            partner_name = partner_names[episode_index % len(partner_names)]
            partner = make_partner(partner_name, seed=seed)

            trace = backend.rollout(
                focal_policy=policy,
                partner_policy=partner,
                layout=layout,
                seed=seed,
                partner_name=partner_name,
                mode="train_rollout",
            )
            train_returns.append(trace.total_reward)

            for step in trace.steps:
                signature = step.focal_observation.signature()
                action_values[signature][step.action_agent_0] += max(trace.total_reward, 0.0) + 0.05

        updated_lookup = dict(package.action_lookup)
        for signature, values in action_values.items():
            updated_lookup[signature] = max(values.items(), key=lambda pair: pair[1])[0]

        candidate = PolicyPackage.from_dict(package.to_dict())
        candidate.action_lookup = updated_lookup
        candidate.metadata["last_iteration_mean_train_return"] = round(mean(train_returns), 4) if train_returns else 0.0
        eval_results = evaluate_package(candidate, config, include_self_play=False)
        candidate_score = summarize_eval(eval_results)["primary_score"]

        iteration_notes.append(
            f"iter {iteration}: train_return={round(mean(train_returns), 4) if train_returns else 0.0}, "
            f"probe_score={candidate_score}, lookup={len(updated_lookup)}"
        )

        if candidate_score >= best_score + float(rlvr_cfg.get("min_improvement_to_keep", 0.0)):
            best_score = float(candidate_score)
            best_package = candidate
            package = candidate

    final_results = evaluate_package(best_package, config, include_self_play=True)
    metrics = summarize_eval(final_results)
    metrics.update(
        {
            "track": track,
            "run_name": run_name,
            "backend": config.get("backend"),
            "benchmark_eligible": bool(config.get("benchmark_eligible", False)),
            "iterations_completed": iterations,
            "num_action_lookup_entries": len(best_package.action_lookup),
        }
    )
    best_package.metadata.update(
        {
            "track": track,
            "algorithm": "rlvr_scaffold",
            "best_probe_score": best_score,
            "iteration_notes": iteration_notes,
        }
    )

    summary = render_summary_markdown(
        track=track,
        run_name=run_name,
        metrics=metrics,
        notes=iteration_notes,
    )
    record_path = write_record_bundle(
        config.get("output_dir", "outputs/rlvr"),
        config=config,
        metrics=metrics,
        episode_rows=[row.to_dict() for row in final_results],
        policy_package=best_package,
        summary_markdown=summary,
    )
    return {
        "record_path": str(record_path),
        "metrics": metrics,
        "policy_package": best_package.to_dict(),
    }
