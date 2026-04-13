from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanocoop.baselines import offline_sft, prompt_opt, rlvr
from nanocoop.envs import make_backend
from nanocoop.evaluation import evaluate_package
from nanocoop.evaluation import package_to_policy
from nanocoop.episode_plan import build_cross_play_episodes, resolve_episode_ids, selected_episode_ids
from nanocoop.io import load_json, load_yaml
from nanocoop.partner_zoo import make_partner
from nanocoop.policy import make_seed_package
from nanocoop.record_bundle import write_record_bundle
from nanocoop.schema import PolicyPackage
from nanocoop.score import render_summary_markdown, run_contract_metadata, summarize_eval


def _load_config(path: str) -> dict[str, Any]:
    return load_yaml(path)


def _cmd_offline(args: argparse.Namespace) -> int:
    result = offline_sft.run(_load_config(args.config))
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


def _cmd_rlvr(args: argparse.Namespace) -> int:
    result = rlvr.run(_load_config(args.config))
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


def _cmd_prompt_opt(args: argparse.Namespace) -> int:
    result = prompt_opt.run(_load_config(args.config))
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


def _cmd_starter_agent(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    package = make_seed_package(
        name=str(config.get("run_name", "starter_agent")),
        backend=str(config.get("backend", "overcookedv2")),
        prompt=str(config.get("policy", {}).get("seed_prompt", "")),
        model_name=str(config.get("model", {}).get("name", "")) or None,
    )
    results = evaluate_package(
        package,
        config,
        include_self_play=not args.no_self_play,
        episode_ids=resolve_episode_ids(args.episodes),
        workers=args.workers,
        progress=True,
    )
    metrics = summarize_eval(results)
    metrics.update(run_contract_metadata(config))
    metrics.update(
        {
            "track": config.get("track", "starter_agent"),
            "run_name": config.get("run_name", "starter_agent"),
            "backend": config.get("backend", "overcookedv2"),
            "benchmark_eligible": bool(config.get("benchmark_eligible", False)),
        }
    )
    package.metadata.update(
        {
            "track": config.get("track", "starter_agent"),
            "algorithm": "nochange_starter_agent",
        }
    )
    output_dir = config.get("output_dir")
    if output_dir:
        notes = ["No-change starter policy package."]
        if metrics.get("failed_episodes"):
            notes.append(
                "Known v0.1 limitation: unresolved wide-layout episodes are retained "
                "as visible baseline failures, not hidden from the score."
            )
        gif_path = None
        if not args.no_gif:
            gif_path = _write_representative_gif(package, config, results, output_dir)
            if gif_path is not None:
                notes.append(f"rollout gif: `{gif_path.name}`")
        summary = render_summary_markdown(
            track=str(config.get("track", "starter_agent")),
            run_name=str(config.get("run_name", "starter_agent")),
            metrics=metrics,
            notes=notes,
        )
        write_record_bundle(
            output_dir,
            config=config,
            metrics=metrics,
            episode_rows=[row.to_dict() for row in results],
            policy_package=package,
            summary_markdown=summary,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _write_representative_gif(
    package: PolicyPackage,
    config: dict[str, Any],
    results,
    output_dir: str,
):
    cross_play = [row for row in results if row.mode == "cross_play"]
    if not cross_play:
        return None
    representative = next((row for row in cross_play if row.success), cross_play[0])
    backend = make_backend(config)
    trace = backend.rollout(
        focal_policy=package_to_policy(
            package, config, rng_seed=representative.episode_id or representative.seed
        ),
        partner_policy=make_partner(representative.partner_name, seed=representative.seed),
        layout=representative.layout,
        seed=representative.seed,
        partner_name=representative.partner_name,
        mode="cross_play",
        capture_states=True,
    )
    gif_name = f"rollout_episode_{representative.episode_id or representative.seed}.gif"
    return backend.write_rollout_gif(trace, Path(output_dir) / gif_name)


def _cmd_eval(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    package = PolicyPackage.from_dict(load_json(args.package))
    results = evaluate_package(
        package,
        config,
        include_self_play=not args.no_self_play,
        episode_ids=resolve_episode_ids(args.episodes),
        workers=args.workers,
        progress=True,
    )
    print(json.dumps(summarize_eval(results), indent=2, sort_keys=True))
    return 0


def _cmd_episodes(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    selected = set(resolve_episode_ids(args.episodes) or selected_episode_ids(config))
    rows = [
        {
            "episode_id": episode.episode_id,
            "selected": episode.episode_id in selected,
            "layout": episode.layout,
            "partner_name": episode.partner_name,
            "seed": episode.seed,
        }
        for episode in build_cross_play_episodes(config)
    ]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanocoop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline_parser = subparsers.add_parser("offline")
    offline_parser.add_argument("--config", required=True)
    offline_parser.set_defaults(func=_cmd_offline)

    rlvr_parser = subparsers.add_parser("rlvr")
    rlvr_parser.add_argument("--config", required=True)
    rlvr_parser.set_defaults(func=_cmd_rlvr)

    prompt_parser = subparsers.add_parser("prompt-opt")
    prompt_parser.add_argument("--config", required=True)
    prompt_parser.set_defaults(func=_cmd_prompt_opt)

    starter_parser = subparsers.add_parser("starter-agent")
    starter_parser.add_argument("--config", required=True)
    starter_parser.add_argument("--no-self-play", action="store_true")
    starter_parser.add_argument("--episodes")
    starter_parser.add_argument("--workers", type=int)
    starter_parser.add_argument("--no-gif", action="store_true")
    starter_parser.set_defaults(func=_cmd_starter_agent)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--package", required=True)
    eval_parser.add_argument("--no-self-play", action="store_true")
    eval_parser.add_argument("--episodes")
    eval_parser.add_argument("--workers", type=int)
    eval_parser.set_defaults(func=_cmd_eval)

    episodes_parser = subparsers.add_parser("episodes")
    episodes_parser.add_argument("--config", required=True)
    episodes_parser.add_argument("--episodes")
    episodes_parser.set_defaults(func=_cmd_episodes)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
