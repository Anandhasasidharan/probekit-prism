FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" \
    HF_HOME=/hf \
    HF_HUB_DISABLE_XET=1

RUN python -m pip install --no-cache-dir uv==0.11.6
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv export --locked --no-dev --no-emit-project --no-header --no-annotate --no-hashes \
      --output-file /tmp/locked-requirements.txt \
    && torch_version="$(sed -nE 's/^torch==([^;[:space:]]+).*$/\1/p' /tmp/locked-requirements.txt)" \
    && test -n "$torch_version" \
    && grep -Ev '^(torch|triton|cuda-[^=]+|nvidia-[^=]+)==' /tmp/locked-requirements.txt \
      > /tmp/cpu-requirements.txt \
    && uv pip install --system --no-cache --no-deps -r /tmp/cpu-requirements.txt \
    && uv pip install --system --no-cache --no-deps \
      --index-url https://download.pytorch.org/whl/cpu "torch==${torch_version}+cpu" \
    && uv pip install --system --no-cache --no-deps . \
    && uv pip check --system

COPY configs ./configs
EXPOSE 8000
CMD ["probekit", "serve", "--config", "configs/benchmark_cpu.yaml", "--serve-config", "configs/serve_docker.yaml"]
