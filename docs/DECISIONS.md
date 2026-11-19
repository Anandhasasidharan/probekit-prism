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

## Phase 2

- A seeded sequence-ID hash reserves the same held-out split for reconstruction and LM evaluation. The normalization scale is fit on training activations and recorded with the checkpoint.
- TopK uses rectified selected values, with the smallest positive float32 normal value for degenerate selections. L0 counts stored nonzeros. Live-feature statistics use a threshold of 1e-8 to exclude numerical placeholders.
- LM evaluation keeps position zero and padding intact when replacing a layer output. It scores non-padding next-token targets. Loss recovered is undefined when zero ablation does not worsen the clean loss.
- JBloom's published SAE for `blocks.9.hook_resid_pre` evaluates the same residual location as the output of zero-based GPT-2 block 8. The published Safetensors weights and config are loaded at a pinned revision. Other releases can use the optional SAE Lens SDK.
- The recorded CPU benchmark uses a 30,000-input-token corpus budget and a smaller dictionary. The default training settings target a larger corpus and dictionary.

## Phase 3

- Live features are sampled with a seeded RNG. Strongest activations are exact within the explanation split. Quantile examples come from a bounded uniform reservoir, so their percentiles are estimates.
- Detection uses a disjoint sequence-ID split and equal numbers of firing and non-firing target tokens where available. Feature examples show the marked token and neighboring context. A feature without both held-out classes has an undefined score.
- The cache uses the checkpoint SHA-256 and stores explanations and scores separately. Increment the prompt version when changing prompts or scoring settings for the same checkpoint.
- The offline provider emits placeholder labels and constant probabilities for smoke testing. The Anthropic-backed path requires `ANTHROPIC_API_KEY`.

## Phase 4

- The PII data is a balanced synthetic benchmark with positive and near-miss email, phone, SSN and card-like formats. It uses reserved or invalid identifiers. The regex labels express format resemblance, and the card label checks Luhn. No VeriAlign middleware or real tool-misuse labels were supplied in this repository.
- The OOD split uses different surrounding templates and changes parts of the value format. Sequence IDs and normalized text are disjoint, and both probe spaces use the same token sequences, pooling, stratified subsets and validation C grid.
- Standardization is fitted on each training subset and folded into the learned logistic coefficients. Validation AUROC selects C and ties favor the smallest C. Test and OOD scores use a fixed probability threshold. Sample-efficiency dispersion is the population standard deviation over the configured seeds.
- Counts describe nonzero learned coefficients. Sparse token activations can still produce dense pooled features and probe weights.
