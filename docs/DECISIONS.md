# MVP decisions

## Phase 0

- The vision document was supplied at `../probekit-v2.md`. The scoped task governs the Python version, supported model and shipped features.
- The MVP uses a single Python 3.11 distribution. Existing workspace packages and deployment files remain in place for historical reference. The prior suite is in `legacy/tests`; its original environment and entry points belong to commit `eb06e37`.
- YAML paths are relative to the YAML file. The model revision is pinned and layer numbers are zero-based.
- WikiText-103 raw is the public corpus default. Local JSONL supports small offline and controlled runs.
