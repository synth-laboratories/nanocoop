from __future__ import annotations

from pathlib import Path
from typing import Any

from nanocoop.io import dump_json, dump_jsonl, dump_yaml, ensure_dir
from nanocoop.schema import PolicyPackage


def write_record_bundle(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    policy_package: PolicyPackage,
    summary_markdown: str,
) -> Path:
    out = ensure_dir(output_dir)
    dump_yaml(out / "config.yaml", config)
    dump_json(out / "metrics.json", metrics)
    dump_json(out / "policy_package.json", policy_package.to_dict())
    dump_jsonl(out / "episode_results.jsonl", episode_rows)
    (out / "summary.md").write_text(summary_markdown, encoding="utf-8")
    return out
