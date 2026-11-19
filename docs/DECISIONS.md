# MVP decisions

## Phase 0

- The vision document was supplied at `../probekit-v2.md`. The scoped task governs the Python version, supported model and shipped features.
- The MVP uses a single Python 3.11 distribution. Existing workspace packages and deployment files remain in place for historical reference. The prior suite is in `legacy/tests`; its original environment and entry points belong to commit `eb06e37`.
- YAML paths are relative to the YAML file. The model revision is pinned and layer numbers are zero-based.
- WikiText-103 raw is the public corpus default. Local JSONL supports small offline and controlled runs.

## Phase 1

- Each written sequence has at least two tokens. GPT-2 receives the first token as causal context, and storage omits position zero because it can dominate the activation distribution. Padding is excluded too.
- The corpus limit counts retained model input tokens, including position zero and any truncation at the limit. Sidecar token IDs are the exact unpadded IDs passed to the model. Its original text may extend past truncation.
- The gate uses random-weight GPT-2 with a local tokenizer so CI requires no downloads. It compares the block hook to the next block's independently returned hidden-state input.
- Model and public corpus revisions are pinned. A local checkpoint can be supplied for offline runs.
