#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec "${PROBEKIT_PYTHON:-python}" "$repo_dir/scripts/run_pipeline.py" "$@"
