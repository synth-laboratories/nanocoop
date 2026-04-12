from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from nanocoop.schema import EpisodeTrace


def filter_teacher_traces(
    traces: Iterable[EpisodeTrace],
    min_return_threshold: float,
) -> list[EpisodeTrace]:
    return [trace for trace in traces if trace.total_reward >= min_return_threshold and trace.success]


def build_action_lookup(
    traces: Iterable[EpisodeTrace],
    *,
    min_votes: int = 1,
    max_examples_per_signature: int = 3,
) -> tuple[dict[str, str], list[dict]]:
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    examples_by_signature: dict[str, list[dict]] = defaultdict(list)

    for trace in traces:
        for step in trace.steps:
            signature = step.focal_observation.signature()
            action = step.action_agent_0
            votes[signature][action] += max(trace.total_reward, 0.0) + 0.01
            if len(examples_by_signature[signature]) < max_examples_per_signature:
                examples_by_signature[signature].append(
                    {
                        "observation": step.focal_observation.to_prompt(),
                        "action": action,
                        "return": trace.total_reward,
                    }
                )

    action_lookup: dict[str, str] = {}
    examples: list[dict] = []

    for signature, action_votes in votes.items():
        total_votes = sum(action_votes.values())
        if total_votes < float(min_votes):
            continue
        best_action = max(action_votes.items(), key=lambda pair: pair[1])[0]
        action_lookup[signature] = best_action
        examples.extend(examples_by_signature[signature])

    return action_lookup, examples


def summarize_dataset(traces: Iterable[EpisodeTrace]) -> dict:
    traces = list(traces)
    if not traces:
        return {
            "num_traces": 0,
            "mean_return": 0.0,
            "success_rate": 0.0,
        }
    return {
        "num_traces": len(traces),
        "mean_return": mean(trace.total_reward for trace in traces),
        "success_rate": mean(1.0 if trace.success else 0.0 for trace in traces),
    }
