# ProbeKit v2 MVP specification

## Pipeline

| Stage | Contract | Validation |
| --- | --- | --- |
| Harvest | Stream pinned public data or local JSONL through GPT-2 Small. Save zstd Parquet with sequence ID, token position, zero-based layer, model version, and float32 residuals. Save unpadded tokens and original text in a sidecar. | Fresh-forward parity on ten sequences, including a local GPT-2 Small checkpoint. |
| SAE | Train a normalized TopK SAE with Adam, decoder-column renormalization, streamed batches, periodic checkpoints, and optional auxiliary dead-feature revival. | Exact sparsity, unit norms, reduced held-out MSE, and identity and zero-splice checks. |
| SAE evaluation | Report L0, reconstruction MSE, fraction of variance explained, dead-feature rate, and LM loss recovered. Optionally score JBloom's pinned SAE or a compatible SAE Lens release. | Recorded held-out CPU run with clean, reconstructed, and zero-ablated LM losses. |
| AutoInterp | Sample live features, collect strong and approximate quantile examples, label with Haiku or a pluggable client, and score held-out target-token detections. Cache by SAE digest, feature ID, explainer model, and prompt version. | Mocked Anthropic client, cache-hit check, disjoint sequence IDs, CSV, and plot. |
| Probes | Fit raw and SAE-latent logistic regressions on pooled sequence features. Validate C on the validation split and report both test and OOD metrics, paired sample-efficiency runs, and nonzero weights. | Separable fixtures, leakage checks, and a recorded GPT-2 Small benchmark. |
| Export and API | Export two probe graphs and the affine SAE encoder. Serve the probe, ranked feature activations with optional labels, and health. | Absolute ONNX parity at 1e-4, FastAPI TestClient coverage, and separate CPU latency measurements. |
| Packaging | Python 3.11 distribution, YAML-configured Typer CLI, CPU Docker Compose service, and a single-process CPU demo runner. | Ruff, offline fast tests in CI, and a marked slow integration test invoking the shell runner. |

## Evaluation and serving conventions

The first position stays in GPT-2's input and is excluded from training
activations, probe pooling, and SAE redirection. A seeded sequence-ID hash
separates training and held-out SAE data. LM losses are token-weighted averages.
Probe scores are sigmoid probabilities with a fixed configurable threshold.
Returned feature activations are the maximum across captured token positions.
The API uses ONNX Runtime's CPU provider and configures GPT-2 independently.

## Deferred capabilities

Feature steering, drift monitoring, a registry API, Kubernetes, Postgres,
Grafana, larger language models, and the async harvester belong to the broader
vision. Real tool-misuse log evaluation requires a labeled dataset.
