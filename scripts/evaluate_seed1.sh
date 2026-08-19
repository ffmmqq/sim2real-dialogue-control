#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m pipeline.evaluation.run_frozen_evaluation \
  --config configs/evaluation_seed1.json \
  --run-name final_seed1 \
  --output-root evaluation_runs \
  "$@"

