#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

DATASET="$DATASET" \
EMB_MODEL="$EMB_MODEL" \
EMBED_STYLE="$EMBED_STYLE" \
EMBED_BACKEND="$EMBED_BACKEND" \
EMBED_QUERY_PREFIX="$EMBED_QUERY_PREFIX" \
EMBED_PASSAGE_PREFIX="$EMBED_PASSAGE_PREFIX" \
ENTITY_NAMES_JSON="$ENTITY_NAMES_JSON" \
bash trm_rag_style/scripts/run_embed.sh
