# syntax=docker/dockerfile:1.7
FROM astral/uv:python3.10-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md Makefile /app/

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --extra overcookedv2 --no-install-project

COPY configs /app/configs
COPY docs /app/docs
COPY scripts /app/scripts
COPY smr /app/smr
COPY src /app/src
COPY submission /app/submission

RUN find /app/scripts -type f -name '*.sh' -exec chmod +x {} +

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --extra overcookedv2

CMD ["bash"]
