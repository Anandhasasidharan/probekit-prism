# ProbeKit v2

Production-grade interpretability infrastructure: harvest activations, train
SAEs, trace circuits via **cross-layer transcoders (CLTs)**, auto-label
features with LLMs, train probes on **three feature spaces** (raw / SAE /
circuit-derived), steer model behavior at inference time, and monitor drift —
all deployable via Docker/Kubernetes.

Built to run end-to-end on a laptop-class GPU (**RTX 3050, ~4–6 GB VRAM**);
everything also runs on CPU.

## Research framing

Goodfire/Rakuten claimed SAE probes beat raw-activation probes; DeepMind's
mech-interp team published negative results around the same time. Rather than
adjudicate that dispute, ProbeKit v2 moves a level up: it reproduces
**Anthropic's circuit-tracing methodology** (Ameisen et al. 2025) at GPT-2
scale and extends the original SAE-vs-raw probe question into a three-way
comparison:

> **Do circuit-level, causally-validated features make better probes than
> either raw activations or SAE latents?**

| | SAEs | CLTs / attribution graphs | Circuit-informed steering | Circuit-level drift |
|---|---|---|---|---|
| SAE Lens | ✅ | ❌ | ❌ | ❌ |
| Goodfire Silico (commercial) | ✅ | proprietary | ✅ feature-level | ✅ |
| Anthropic published method | N/A | ✅ originates here | research only | N/A |
| **ProbeKit v2** | ✅ baseline | ✅ | ✅ circuit-level | ✅ |

## Quickstart

```bash
uv sync --all-packages --all-groups
uv run pytest tests/          # 48 tests, offline, uses tiny random models
uv run ruff check .           # lint
uv run mypy packages/         # type check
```

### Full pipeline on a real model (GPT-2 Small, GPU)

```bash
# 1. Harvest activations (GPU-accelerated forward passes)
uv run probekit harvest --prompts data/examples/prompts.jsonl \
    --output data/activations/harvest.parquet

# 2. Train a TopK SAE on the harvested activations + export ONNX
uv run probekit train-sae --data data/activations/harvest.parquet \
    --out data/saes/sae

# 3. Auto-label features with an LLM (set OPENAI_API_KEY; provider=mock works offline)
uv run probekit auto-interp --data data/activations/harvest.parquet \
    --sae data/saes/sae --top-features 10

# 4. Train raw/SAE/circuit probes for a concept and compare
uv run probekit train-probe --concept-file data/examples/pii_detection.jsonl

# 5. Steer generation
uv run probekit steer --prompt "delete all files in /data" \
    --strategy suppress_risky --feature-id 42

# 6. Drift check / serve the API
uv run probekit monitor
uv run probekit serve          # http://localhost:8000/docs
```

## Architecture

```
probekit/
├── packages/
│   ├── core/          # Shared types, Pydantic config, CUDA device utilities
│   ├── harvest/       # Batched activation collection + provenance + Parquet/zstd
│   ├── sae/           # TopK / Gated / JumpReLU / ReLU SAEs + ONNX export
│   ├── circuit/       # CLTs, replacement model, attribution graphs, interpretation
│   ├── auto-interp/   # Example collection → LLM explainer → detection/fuzz scores
│   ├── probes/        # Raw vs. SAE vs. circuit probes, comparator, ONNX export
│   ├── steering/      # Activation interventions, probe-driven controller
│   ├── registry/      # Feature/circuit catalog (SQLite) with stats history
│   ├── monitor/       # Frequency / probe-accuracy / KL drift detectors + alerts
│   └── api/           # FastAPI app + CLI (`probekit ...`)
├── tests/             # 48 tests incl. full-pipeline integration test
├── deploy/            # Dockerfile(s), docker-compose, K8s, Prometheus rules
├── data/              # activations / saes / features / probes / examples
└── config.yaml        # Global configuration
```

### How circuit tracing works here

1. **Collect MLP pairs** (`circuit.collect`) — MLP input/output pairs from any
   HF causal LM supervise transcoder training.
2. **Train cross-layer transcoders** (`circuit.train`) — one small
   encoder→decoder per layer; on GPT-2 Small they reconstruct MLP outputs to
   near-zero MSE in seconds on an RTX 3050.
3. **Replacement model** (`circuit.replacement`) — swaps covered MLPs for sums
   of upstream transcoder writes, making feature interactions linear for a
   given input (frozen-input scheme, documented approximation).
4. **Attribution graphs** (`circuit.attribution`) — node attributions come
   from exact local gradients ∂logit/∂a_f via one backward pass; edge weights
   use Anthropic's virtual-weight approximation; pruned by percentile
   thresholds.
5. **Automated interpretation** (`circuit.interpret`) — structured graph
   summaries + optional LLM narration (Hanna & Ameisen-style probe prompting).

Measured on RTX 3050 Laptop (6 GB): GPT-2 harvest ≈ 1.5 s/batch, CLT training
(4 layers × 1024 features) ≈ 1.5 s, attribution graph (170 nodes / 1.8k edges)
≈ instant, peak VRAM ≈ 0.6 GB.

## Configuration

All knobs live in `config.yaml` (models, capture layers, SAE architecture,
CLT sizes, probe concepts, steering strategies, thresholds). `PROBEKIT_CONFIG`
env var overrides the path; every loader falls back to sane defaults.

## Deployment

```bash
cd deploy
docker compose up -d                 # API + Prometheus (+ Postgres)
docker compose --profile harvester up probekit-harvester
kubectl apply -f kubernetes/
```

Endpoints: `/health`, `/ready`, `/probe/score`, `/features/search`,
`/features/{id}/verify`, `/features/drift`, `/steer/check_and_steer`,
`/monitor/drift`. Rate limiting and request logging included.

## Sources

- Anthropic — *Circuit Tracing: Revealing Computational Graphs in Language
  Models* (Transformer Circuits Thread, 2025); Ameisen et al., cross-layer
  transcoders
- Hanna & Ameisen (2026) — automated attribution-graph interpretation
- Goodfire + Rakuten — SAE Probes for PII Detection (Oct 2025)
- GDM Mechanistic Interpretability Team — negative results on SAEs for
  downstream tasks (2025)
- EleutherAI Delphi — open-source automated interpretability (2025)
- Gao et al. (2024) TopK SAEs; Rajamanoharan et al. (2024) Gated SAEs;
  Decu et al. (2024) JumpReLU SAEs
