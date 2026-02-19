#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/env.sh"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${DATASET}_${MODEL_IMPL}_full_${RUN_TS}}"

# Baselines from env.sh (used to detect "not user-overridden" values)
DEFAULT_WANDB_RUN_NAME_PHASE1="${DATASET}_phase1_paperstyle_ep${EPOCHS_PHASE1}"
DEFAULT_WANDB_RUN_NAME_PHASE2="${DATASET}_phase2_rl_from_phase1_ep${EPOCHS_PHASE2}"
DEFAULT_CKPT_DIR_PHASE1="trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_phase1_paperstyle"
DEFAULT_CKPT_DIR_PHASE2="trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_phase2_rl_from_phase1"
DEFAULT_RESULTS_DIR="experiments/paperstyle_rl/results"

# W&B online by default for full runs.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
export WANDB_ENTITY="${WANDB_ENTITY:-heewon6205-chung-ang-university}"
if [ "${WANDB_RUN_NAME_PHASE1:-}" = "" ] || [ "$WANDB_RUN_NAME_PHASE1" = "$DEFAULT_WANDB_RUN_NAME_PHASE1" ]; then
  export WANDB_RUN_NAME_PHASE1="${RUN_TAG}_phase1"
fi
if [ "${WANDB_RUN_NAME_PHASE2:-}" = "" ] || [ "$WANDB_RUN_NAME_PHASE2" = "$DEFAULT_WANDB_RUN_NAME_PHASE2" ]; then
  export WANDB_RUN_NAME_PHASE2="${RUN_TAG}_phase2_rl"
fi

# Isolate artifacts per run to avoid mixing with previous checkpoints.
if [ "${CKPT_DIR_PHASE1:-}" = "" ] || [ "$CKPT_DIR_PHASE1" = "$DEFAULT_CKPT_DIR_PHASE1" ]; then
  export CKPT_DIR_PHASE1="trm_rag_style/ckpt/${RUN_TAG}/phase1"
fi
if [ "${CKPT_DIR_PHASE2:-}" = "" ] || [ "$CKPT_DIR_PHASE2" = "$DEFAULT_CKPT_DIR_PHASE2" ]; then
  export CKPT_DIR_PHASE2="trm_rag_style/ckpt/${RUN_TAG}/phase2"
fi
if [ "${RESULTS_DIR:-}" = "" ] || [ "$RESULTS_DIR" = "$DEFAULT_RESULTS_DIR" ]; then
  export RESULTS_DIR="experiments/paperstyle_rl/results/${RUN_TAG}"
fi

mkdir -p "$CKPT_DIR_PHASE1" "$CKPT_DIR_PHASE2" "$RESULTS_DIR"

echo "[run_all_wandb] RUN_TAG=$RUN_TAG"
echo "[run_all_wandb] WANDB_MODE=$WANDB_MODE PROJECT=$WANDB_PROJECT ENTITY=$WANDB_ENTITY"
echo "[run_all_wandb] PHASE1_RUN=$WANDB_RUN_NAME_PHASE1"
echo "[run_all_wandb] PHASE2_RUN=$WANDB_RUN_NAME_PHASE2"
echo "[run_all_wandb] CKPT_DIR_PHASE1=$CKPT_DIR_PHASE1"
echo "[run_all_wandb] CKPT_DIR_PHASE2=$CKPT_DIR_PHASE2"
echo "[run_all_wandb] RESULTS_DIR=$RESULTS_DIR"

bash "$SCRIPT_DIR/run_all.sh"
