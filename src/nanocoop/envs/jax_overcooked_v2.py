from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JaxOvercookedV2Backend:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            import jax  # noqa: F401
            import jaxmarl  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency-gated path
            raise RuntimeError(
                "The JAX OvercookedV2 backend requires the optional 'overcookedv2' "
                "dependencies. Install with: pip install -e '.[overcookedv2]'"
            ) from exc

    def rollout(self, *args, **kwargs):  # pragma: no cover - adapter placeholder
        raise NotImplementedError(
            "The official OvercookedV2 adapter interface is scaffolded but intentionally "
            "left minimal here. Extend this class to map JaxMARL observations into the "
            "symbolic observation contract used by NanoCoop."
        )
