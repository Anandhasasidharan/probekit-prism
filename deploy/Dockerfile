# ProbeKit API server image
FROM python:3.12-slim AS base

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY packages ./packages

RUN uv sync --frozen --all-packages --no-dev --compile-bytecode

ENV PATH="/app/.venv/bin:$PATH"
ENV PROBEKIT_CONFIG=/app/config.yaml

COPY config.yaml ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "probekit.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
