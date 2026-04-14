from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_OFFICIAL_EVAL_EPISODE_IDS = [1, 3, 4, 5, 6, 8, 10, 12, 13, 14, 19, 21, 24, 26, 30, 33, 36, 39, 44, 47]
_PUBLIC_TRAIN_EPISODE_IDS = [episode_id for episode_id in range(1, 49) if episode_id not in set(_OFFICIAL_EVAL_EPISODE_IDS)]
PUBLICATION_SMOKE_NOTE = (
    "Publication smoke note: keep the cheap gpt-4.1-nano baseline intact, "
    "prefer complementary roles, and avoid duplicate work."
)


def _parse_episode_ids_override(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _official_eval_episode_ids() -> list[int]:
    raw = str(os.environ.get("NANOCOOP_SUBMISSION_EVAL_EPISODE_IDS", "")).strip()
    if raw:
        parsed = _parse_episode_ids_override(raw)
        if parsed:
            return parsed
    return list(_OFFICIAL_EVAL_EPISODE_IDS)


def _public_train_episode_ids(eval_episode_ids: list[int]) -> list[int]:
    eval_set = set(int(value) for value in eval_episode_ids)
    return [episode_id for episode_id in range(1, 49) if episode_id not in eval_set]


def define() -> dict[str, Any]:
    eval_episode_ids = _official_eval_episode_ids()
    return {
        "name": "gpt41_nano_submission",
        "description": "Single-file NanoCoop submission surface for starter-agent style coordination changes.",
        "train_episode_ids": _public_train_episode_ids(eval_episode_ids),
        "model": {
            "name": "gpt-4.1-nano",
            "api_base": "https://api.openai.com/v1",
            "temperature": 0.0,
            "max_tokens": 256,
            "plan_horizon": 4,
        },
        "policy": {
            "seed_prompt": (
                "You are a cooperative cooking agent.\n"
                "Complement your partner.\n"
                "Prefer complementary roles.\n"
                "Avoid duplicate work.\n"
                "Finish the soup.\n"
                f"{PUBLICATION_SMOKE_NOTE}"
            ),
            "coordination_rules": [
                "Prefer role splits over mirrored movement.",
                "Hand off dishes and ingredients when it shortens the critical path.",
                "Replan when the partner claims your current target.",
            ],
            "action_filters": [
                "Avoid blocking the partner in corridors unless serving immediately.",
                "Do not idle beside the same object your partner is already handling.",
            ],
        },
        "env": {
            "eval_layouts": ["test_time_simple", "test_time_wide", "demo_cook_wide"],
            "max_steps": 80,
            "policy_decision_interval": 8,
            "stochasticity": 0.12,
        },
        "partner_zoo": {"eval": ["courier", "potter", "handoff", "noisy"]},
        "eval": {
            "episode_seed_start": 1,
            "episodes_per_combo": 4,
            "default_episode_count": 20,
            "default_episode_sample_seed": 20260412,
            "workers": 4,
            "episode_ids": eval_episode_ids,
        },
    }


def train(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "define": define(),
        "train_data_dir": str(data_dir),
        "trained": False,
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_episode_ids(data_dir: Path, config: dict[str, Any]) -> list[int]:
    episode_ids_path = data_dir / "episode_ids.json"
    if episode_ids_path.exists():
        payload = json.loads(episode_ids_path.read_text(encoding="utf-8"))
        values = payload.get("episode_ids") if isinstance(payload, dict) else payload
        if isinstance(values, list):
            return [int(item) for item in values]
    raw = config.get("eval", {}).get("episode_ids", [])
    return [int(item) for item in raw]


def _materialize_runner_config(base_config: dict[str, Any], *, episode_ids: list[int], output_dir: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config.update(
        {
            "track": "submission_agent",
            "run_name": "submission_agent_eval",
            "backend": "overcookedv2",
            "benchmark_eligible": False,
            "output_dir": str(output_dir),
        }
    )
    config.setdefault("eval", {})
    config["eval"]["episode_ids"] = episode_ids
    return config


def eval(checkpoint_dir: Path, data_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {"define": define()}
    )
    config = checkpoint.get("define") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict):
        config = define()
    episode_ids = _resolve_episode_ids(data_dir, config)
    record_dir = out_dir / "record"
    runner_config = _materialize_runner_config(config, episode_ids=episode_ids, output_dir=record_dir)
    with tempfile.TemporaryDirectory(prefix="nanocoop-submission-") as temp_dir:
        config_path = Path(temp_dir) / "agent_eval.yaml"
        config_path.write_text(yaml.safe_dump(runner_config, sort_keys=False), encoding="utf-8")
        command = [
            "uv",
            "run",
            "--project",
            str(REPO_ROOT),
            "nanocoop",
            "starter-agent",
            "--config",
            str(config_path),
            "--no-self-play",
            "--workers",
            str(int(runner_config.get("eval", {}).get("workers", 4))),
        ]
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "nanocoop starter-agent eval failed")
    metrics = json.loads(process.stdout)
    result = {
        "primary_score": metrics.get("primary_score"),
        "mean_reward": metrics.get("cross_play_mean_reward"),
        "metrics": metrics,
        "episode_ids": episode_ids,
        "checkpoint": checkpoint,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["define", "train", "eval"])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "out")
    args = parser.parse_args()
    if args.phase == "define":
        print(json.dumps(define(), indent=2, sort_keys=True))
        return 0
    if args.phase == "train":
        train(args.data_dir, args.out_dir)
        return 0
    print(json.dumps(eval(args.checkpoint_dir, args.data_dir, args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
