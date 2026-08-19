#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for seed in 1 2 3 4 5; do
  python -m pipeline.train_seed "$seed" --config configs/training.json "$@"
done

