from __future__ import annotations

from typing import Any

from nanocoop.data import build_action_lookup, filter_teacher_traces, summarize_dataset
from nanocoop.envs import make_backend
from nanocoop.evaluation import evaluate_package
from nanocoop.partner_zoo import make_partner
from nanocoop.policy import OracleTeacherPolicy, make_seed_package
from nanocoop.record_bundle import write_record_bundle
from nanocoop.score import render_summary_markdown, summarize_eval


def run(config: dict[str, Any]) -> dict[str, Any]:
    backend = make_backend(config)
    env_cfg = config.get("env", {})
    offline_cfg = config.get("offline", {})

    train_layouts = list(env_cfg.get("train_layouts", []))
    train_seeds = list(env_cfg.get("train_seeds", []))
    partner_names = list(config.get("partner_zoo", {}).get("train", []))

    teacher = OracleTeacherPolicy()
    teacher_traces = []

    for layout in train_layouts:
        for seed in train_seeds:
            for partner_name in partner_names:
                for episode_idx in range(int(offline_cfg.get("teacher_episodes_per_pair", 4))):
                    partner = make_partner(partner_name, seed=seed + episode_idx)
                    trace = backend.rollout(
                        focal_policy=teacher,
                        partner_policy=partner,
                        layout=layout,
                        seed=seed + episode_idx,
                        partner_name=partner_name,
                        mode="teacher_collect",
                    )
                    teacher_traces.append(trace)

    filtered = filter_teacher_traces(
        teacher_traces,
        min_return_threshold=float(offline_cfg.get("min_return_threshold", 1.0)),
    )
    action_lookup, fewshot_examples = build_action_lookup(
        filtered,
        min_votes=int(offline_cfg.get("min_votes", 1)),
        max_examples_per_signature=int(offline_cfg.get("max_examples_per_signature", 3)),
    )

    seed_prompt = str(config.get("policy", {}).get("seed_prompt", ""))
    package = make_seed_package(
        name=str(config.get("run_name", "offline_run")),
        backend=str(config.get("backend", "mock")),
        prompt=seed_prompt,
        model_name=str(config.get("model", {}).get("name", "")) or None,
    )
    package.action_lookup = action_lookup
    package.fewshot_examples = fewshot_examples[:64]
    package.metadata.update(
        {
            "track": config.get("track"),
            "algorithm": "offline_sft_scaffold",
            "dataset_summary": summarize_dataset(filtered),
            "num_teacher_traces": len(teacher_traces),
            "num_filtered_traces": len(filtered),
        }
    )

    eval_results = evaluate_package(package, config, include_self_play=True)
    metrics = summarize_eval(eval_results)
    metrics.update(
        {
            "track": config.get("track"),
            "run_name": config.get("run_name"),
            "backend": config.get("backend"),
            "benchmark_eligible": bool(config.get("benchmark_eligible", False)),
            "num_teacher_traces": len(teacher_traces),
            "num_filtered_traces": len(filtered),
            "num_action_lookup_entries": len(action_lookup),
        }
    )
    summary = render_summary_markdown(
        track=str(config.get("track")),
        run_name=str(config.get("run_name")),
        metrics=metrics,
        notes=[
            f"teacher traces: {len(teacher_traces)}",
            f"filtered traces: {len(filtered)}",
            f"lookup entries: {len(action_lookup)}",
        ],
    )
    record_path = write_record_bundle(
        config.get("output_dir", "outputs/offline"),
        config=config,
        metrics=metrics,
        episode_rows=[row.to_dict() for row in eval_results],
        policy_package=package,
        summary_markdown=summary,
    )
    return {
        "record_path": str(record_path),
        "metrics": metrics,
        "policy_package": package.to_dict(),
    }
