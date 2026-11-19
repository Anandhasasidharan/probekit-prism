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

## Measured SAE reconstruction

The CPU benchmark used GPT-2 Small at layer 8 and a 30,000-input-token prefix of
WikiText-103. The TopK checkpoint used a 1,024-feature dictionary. The published
JBloom SAE was evaluated directly on the Hugging Face residual stream at the
equivalent hook location. Differences in training corpora, dictionary size,
and TransformerLens residual conventions limit architectural comparisons.

| SAE | L0 | Reconstruction MSE | Variance explained | Dead features | LM loss recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained TopK | 32.00 | 6.216 | 0.370 | 6.74% | 84.06% |
| JBloom pretrained | 270.32 | 30.259 | -2.067 | 0.24% | 94.78% |

See the saved [harvest](runs/harvest.json), [training](runs/sae-training.json),
and [held-out evaluation](runs/sae-metrics.json) reports.
