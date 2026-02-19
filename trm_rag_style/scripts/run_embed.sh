#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
EMBED_DEVICE=${EMBED_DEVICE:-cuda}
EMBED_GPUS=${EMBED_GPUS:-}
# Default to the GNN-RAG-compatible embedding path.
EMBED_STYLE=${EMBED_STYLE:-gnn_rag}
EMBED_BACKEND=${EMBED_BACKEND:-sentence_transformers}
EMBED_QUERY_PREFIX=${EMBED_QUERY_PREFIX:-query: }
EMBED_PASSAGE_PREFIX=${EMBED_PASSAGE_PREFIX:-passage: }
ENTITY_NAMES_JSON=${ENTITY_NAMES_JSON:-data/data/entities_names.json}
OVR=(embed_device="$EMBED_DEVICE" embed_gpus="$EMBED_GPUS")
if [ -n "$EMBED_STYLE" ]; then
  OVR+=(embed_style="$EMBED_STYLE")
fi
if [ -n "$EMBED_BACKEND" ]; then
  OVR+=(embed_backend="$EMBED_BACKEND")
fi
if [ -n "$EMBED_QUERY_PREFIX" ]; then
  OVR+=(embed_query_prefix="$EMBED_QUERY_PREFIX")
fi
if [ -n "$EMBED_PASSAGE_PREFIX" ]; then
  OVR+=(embed_passage_prefix="$EMBED_PASSAGE_PREFIX")
fi
if [ -n "$ENTITY_NAMES_JSON" ]; then
  OVR+=(entity_names_json="$ENTITY_NAMES_JSON")
fi
$PYTHON_BIN -m trm_agent.run --dataset "$DATASET" --stage embed --embedding_model "$EMB_MODEL" \
  --override "${OVR[@]}"
