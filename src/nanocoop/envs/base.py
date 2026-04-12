from __future__ import annotations

from typing import Any

from nanocoop.envs.jax_overcooked_v2 import JaxOvercookedV2Backend


def make_backend(config: dict[str, Any]):
    backend_name = str(config.get("backend", "overcookedv2")).lower()
    if backend_name in {"jax", "jax_overcooked_v2", "overcookedv2"}:
        return JaxOvercookedV2Backend(config=config)
    raise ValueError(f"Unsupported backend: {backend_name}")
