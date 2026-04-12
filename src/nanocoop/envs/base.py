from __future__ import annotations

from typing import Any

from nanocoop.envs.jax_overcooked_v2 import JaxOvercookedV2Backend
from nanocoop.envs.mock_overcooked_v2 import MockOvercookedV2Backend


def make_backend(config: dict[str, Any]):
    backend_name = str(config.get("backend", "mock")).lower()
    env_cfg = config.get("env", {})
    if backend_name == "mock":
        return MockOvercookedV2Backend(
            max_steps=int(env_cfg.get("max_steps", 8)),
            stochasticity=float(env_cfg.get("stochasticity", 0.1)),
        )
    if backend_name in {"jax", "jax_overcooked_v2", "overcookedv2"}:
        return JaxOvercookedV2Backend(config=config)
    raise ValueError(f"Unsupported backend: {backend_name}")
