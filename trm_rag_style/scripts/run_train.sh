#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
EVAL_LIMIT=${EVAL_LIMIT:-200}
DEBUG_EVAL_N=${DEBUG_EVAL_N:-5}
EVAL_EVERY_EPOCHS=${EVAL_EVERY_EPOCHS:-2}
EVAL_START_EPOCH=${EVAL_START_EPOCH:-5}
WANDB_MODE=${WANDB_MODE:-disabled}
WANDB_PROJECT=${WANDB_PROJECT:-graph-traverse}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

TORCHRUN=${TORCHRUN:-torchrun}
$TORCHRUN --nproc_per_node=3 --master_port=29500 -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --embedding_model "$EMB_MODEL" \
  --stage train \
  --override \
    eval_limit="$EVAL_LIMIT" \
    debug_eval_n="$DEBUG_EVAL_N" \
    eval_every_epochs="$EVAL_EVERY_EPOCHS" \
    eval_start_epoch="$EVAL_START_EPOCH" \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    wandb_entity="$WANDB_ENTITY" \
    wandb_run_name="$WANDB_RUN_NAME"
