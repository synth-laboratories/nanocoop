from __future__ import annotations

from typing import Any

from nanocoop.evaluation import evaluate_package
from nanocoop.policy import make_seed_package
from nanocoop.record_bundle import write_record_bundle
from nanocoop.score import render_summary_markdown, summarize_eval


def run(config: dict[str, Any]) -> dict[str, Any]:
    prompt_cfg = config.get("prompt_opt", {})
    seed_prompt = str(config.get("policy", {}).get("seed_prompt", "")).strip()
    clauses = list(prompt_cfg.get("clauses", []))
    max_candidates = min(int(prompt_cfg.get("max_candidates", len(clauses) + 1)), len(clauses) + 1)

    candidate_prompts = [seed_prompt]
    current = seed_prompt
    for clause in clauses:
        current = f"{current.rstrip()}\n{clause}".strip()
        candidate_prompts.append(current)
        if len(candidate_prompts) >= max_candidates:
            break

    probe_config = dict(config)
    probe_env = dict(config.get("env", {}))
    probe_env["eval_seeds"] = list(config.get("env", {}).get("train_seeds", []))
    probe_env["eval_layouts"] = list(config.get("env", {}).get("train_layouts", []))
    probe_config["env"] = probe_env
    probe_config["partner_zoo"] = {"eval": list(config.get("partner_zoo", {}).get("train", []))}

    best_package = None
    best_score = float("-inf")
    candidate_notes: list[str] = []

    for idx, prompt in enumerate(candidate_prompts):
        package = make_seed_package(
            name=f"{config.get('run_name', 'prompt_opt')}_cand_{idx}",
            backend=str(config.get("backend", "overcookedv2")),
            prompt=prompt,
            model_name=str(config.get("model", {}).get("name", "")) or None,
        )
        probe_results = evaluate_package(package, probe_config, include_self_play=False)
        probe_score = summarize_eval(probe_results)["primary_score"]
        candidate_notes.append(f"candidate {idx}: probe_score={probe_score} flags={package.behavior_flags}")
        if probe_score > best_score:
            best_score = float(probe_score)
            best_package = package

    assert best_package is not None

    final_results = evaluate_package(best_package, config, include_self_play=True)
    metrics = summarize_eval(final_results)
    metrics.update(
        {
            "track": config.get("track"),
            "run_name": config.get("run_name"),
            "backend": config.get("backend"),
            "benchmark_eligible": bool(config.get("benchmark_eligible", False)),
            "num_candidates": len(candidate_prompts),
        }
    )
    best_package.metadata.update(
        {
            "track": config.get("track"),
            "algorithm": "prompt_opt_scaffold",
            "best_probe_score": best_score,
            "candidate_notes": candidate_notes,
        }
    )

    summary = render_summary_markdown(
        track=str(config.get("track")),
        run_name=str(config.get("run_name")),
        metrics=metrics,
        notes=candidate_notes,
    )
    record_path = write_record_bundle(
        config.get("output_dir", "outputs/prompt_opt"),
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
