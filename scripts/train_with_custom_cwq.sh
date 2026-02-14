#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_preprocess_custom_cwq.sh"
bash "$SCRIPT_DIR/run_embed_custom_cwq.sh"
bash "$SCRIPT_DIR/run_train_custom_cwq.sh"
