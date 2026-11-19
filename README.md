# ProbeKit v2 MVP

ProbeKit is a research-to-deployment pipeline for comparing raw GPT-2 residual
stream probes with probes on sparse autoencoder latents.

## Setup

Install Python 3.11 and uv, then run:

```bash
uv sync --locked
uv run probekit harvest --config configs/default.yaml
uv run pytest -m "not slow"
```

The default config streams a pinned WikiText-103 subset and runs GPT-2 Small on
CUDA when available, with CPU fallback. To use local data, set `corpus.jsonl`
to a JSONL file with a `text` field on every record and omit `corpus.dataset`.
Model and data paths in YAML are relative to the YAML file.

The harvester writes zstd-compressed residual activations from the configured
zero-based GPT-2 block and a sidecar with each sequence's text and exact input
token IDs. Position zero stays in the model context but is omitted from the
activation data because it can behave as an attention sink. Padding is omitted.

See [implementation decisions](docs/DECISIONS.md) for evaluation and
reproducibility choices.
