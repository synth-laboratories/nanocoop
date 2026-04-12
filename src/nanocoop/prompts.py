from __future__ import annotations

from typing import Iterable

BEHAVIOR_KEYWORDS = {
    "share hidden info early": "share_hidden_info_early",
    "complement your partner": "complement_partner",
    "avoid duplicate work": "avoid_duplicate_work",
    "finish the soup": "finish_pipeline",
    "recover quickly after stochastic failures": "recover_from_failures",
    "infer your partner's convention": "infer_partner_convention",
    "prefer complementary roles": "prefer_complementary_roles",
}


def extract_behavior_flags(prompt: str) -> list[str]:
    lowered = prompt.lower()
    flags = [flag for phrase, flag in BEHAVIOR_KEYWORDS.items() if phrase in lowered]
    if "share hidden info" in lowered and "share_hidden_info_early" not in flags:
        flags.append("share_hidden_info_early")
    return sorted(set(flags))


def render_fewshot_examples(examples: Iterable[dict]) -> str:
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        obs = example.get("observation", "")
        action = example.get("action", "")
        blocks.append(
            f"Example {index}\nObservation:\n{obs}\nCorrect action: {action}"
        )
    return "\n\n".join(blocks)
