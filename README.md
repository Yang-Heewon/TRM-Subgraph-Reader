# GRAPH-TRAVERSE

This repository is trimmed to the active training pipeline only.

## Active Pipeline

- `experiments/paperstyle_rl/*`
- `trm_rag_style/scripts/*` (only scripts used by the pipeline)
- `scripts/setup_and_preprocess.sh`
- `scripts/download_data.sh`
- `scripts/lib/portable_env.sh`
- core Python modules: `trm_agent`, `trm_rag_style`, `trm_unified`

## One-Command Run

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
bash experiments/paperstyle_rl/run_all_wandb.sh
```

This runs:

1. `00_preprocess.sh`
2. `01_embed.sh`
3. `02_train_phase1.sh`
4. `03_train_phase2_rl.sh`
5. `04_eval_phase2_test.sh`

## Step-by-Step

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE

# 1) preprocess
bash experiments/paperstyle_rl/00_preprocess.sh

# 2) embed (GNN-RAG gnn exact style)
bash experiments/paperstyle_rl/01_embed.sh

# 3) phase1 supervised
bash experiments/paperstyle_rl/02_train_phase1.sh

# 4) phase2 RL from latest phase1 checkpoint
bash experiments/paperstyle_rl/03_train_phase2_rl.sh

# 5) final test eval
bash experiments/paperstyle_rl/04_eval_phase2_test.sh
```

## Common Overrides

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE

ENDPOINT_LOSS_MODE=entity_dist_main \
RELATION_AUX_WEIGHT=0.2 \
EPOCHS_PHASE1=10 \
EVAL_LIMIT=200 \
bash experiments/paperstyle_rl/02_train_phase1.sh
```

## Notes

- `EMBED_STYLE=gnn_rag_gnn_exact` is configured as default in `experiments/paperstyle_rl/env.sh`.
- In exact mode, query/passage prefixes are forced to empty.
- `DATASET=all` preprocess path was removed in this slim pipeline. Use `DATASET=cwq` or `DATASET=webqsp`.
