#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MAX_STEPS="${MAX_STEPS:-4}"
MAX_PATHS="${MAX_PATHS:-4}"
MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"

echo "[step] preprocess cwq first (BFS depth=$MAX_STEPS, max_paths=$MAX_PATHS)"
python -m trm_rag_style.run \
  --dataset cwq \
  --stage preprocess \
  --override \
    max_steps="$MAX_STEPS" \
    max_paths="$MAX_PATHS" \
    mine_max_neighbors="$MINE_MAX_NEIGHBORS"

echo "[step] preprocess webqsp"
python -m trm_rag_style.run \
  --dataset webqsp \
  --stage preprocess \
  --override \
    max_steps="$MAX_STEPS" \
    max_paths="$MAX_PATHS" \
    mine_max_neighbors="$MINE_MAX_NEIGHBORS"

echo "[done] cwq -> webqsp preprocess complete"
