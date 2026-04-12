from __future__ import annotations

import argparse
import json
from typing import Any

from nanocoop.baselines import offline_sft, prompt_opt, rlvr
from nanocoop.io import load_json, load_yaml
from nanocoop.schema import PolicyPackage
from nanocoop.evaluation import evaluate_package
from nanocoop.score import summarize_eval


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


def _cmd_eval(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    package = PolicyPackage.from_dict(load_json(args.package))
    results = evaluate_package(package, config, include_self_play=not args.no_self_play)
    print(json.dumps(summarize_eval(results), indent=2, sort_keys=True))
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

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--package", required=True)
    eval_parser.add_argument("--no-self-play", action="store_true")
    eval_parser.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
