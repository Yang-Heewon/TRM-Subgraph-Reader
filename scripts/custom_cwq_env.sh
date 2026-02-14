#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

TRAIN_JSONL="${TRAIN_JSONL:-/data2/workspace/heewon/GRAPH-TRAVERSE/data/train_split.jsonl}"
DEFAULT_DEV="/data2/workspace/heewon/GRAPH-TRAVERSE/data/test_split.jsonl"
if [[ -f "$DEFAULT_DEV" ]]; then
  DEV_JSONL="${DEV_JSONL:-$DEFAULT_DEV}"
else
  DEV_JSONL="${DEV_JSONL:-$TRAIN_JSONL}"
fi

ENTITIES_TXT="${ENTITIES_TXT:-/data2/workspace/heewon/논문작업/data/CWQ/entities.txt}"
RELATIONS_TXT="${RELATIONS_TXT:-/data2/workspace/heewon/논문작업/data/CWQ/relations.txt}"
ENTITY_NAMES_JSON="${ENTITY_NAMES_JSON:-/data2/workspace/heewon/논문작업/data/entities_names.json}"

MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"     # local-hash or HF model name
EMB_TAG="${EMB_TAG:-cwq_custom}"
RUN_ID="${RUN_ID:-cwq_custom}"

PROC_DIR="$ROOT_DIR/trm_rag_style/processed/${RUN_ID}"
EMB_DIR="$ROOT_DIR/trm_rag_style/emb/${RUN_ID}_${EMB_TAG}"
CKPT_DIR="$ROOT_DIR/trm_rag_style/ckpt/${RUN_ID}_${MODEL_IMPL}"
ENTITY_TEXT_OUT="$ROOT_DIR/data/${RUN_ID}_entity_text.txt"
