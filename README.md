# GRAPH-TRAVERSE

This repository contains a slimmed training flow for TRM-based KG traversal.

## Scope

Only the active pipeline is kept:

- `experiments/paperstyle_rl/*` (main entrypoint)
- `trm_rag_style/scripts/*` (runtime wrappers used by the pipeline)
- `scripts/download_data.sh`
- `scripts/prepare_rog_hf_data.py`
- `scripts/lib/portable_env.sh`
- core modules: `trm_agent`, `trm_rag_style`, `trm_unified`

## Quick Start

Cross-platform (recommended):

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
python experiments/paperstyle_rl/run_pipeline.py --stage all
```

Linux bash wrappers:

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
bash experiments/paperstyle_rl/run_all_wandb.sh
```

This executes download, preprocess, embedding, phase1 training, phase2 RL training, and test evaluation.

## Step-by-Step

Cross-platform:

```bash
python experiments/paperstyle_rl/run_pipeline.py --stage embed
python experiments/paperstyle_rl/run_pipeline.py --stage phase1
python experiments/paperstyle_rl/run_pipeline.py --stage phase2
python experiments/paperstyle_rl/run_pipeline.py --stage test
```

Linux bash:

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
bash experiments/paperstyle_rl/01_embed.sh
bash experiments/paperstyle_rl/02_train_phase1.sh
bash experiments/paperstyle_rl/03_train_phase2_rl.sh
bash experiments/paperstyle_rl/04_eval_phase2_test.sh
```

`01_embed.sh` now performs:
1. RoG HF download/convert (`scripts/download_data.sh`)
2. preprocess
3. embed

## Embed-Only HF Command

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
bash trm_rag_style/scripts/run_embed.sh
```

Key env vars for `run_embed.sh`:
- `DATASET` (`cwq` or `webqsp`)
- `EMB_MODEL`
- `EMB_TAG`
- `EMBED_STYLE`
- `EMBED_BACKEND`
- `DOWNLOAD_FIRST` (`1` by default)
- `RUN_PREPROCESS` (`1` by default)

## Common Override Example

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
ENDPOINT_LOSS_MODE=entity_dist_main \
RELATION_AUX_WEIGHT=0.2 \
EPOCHS_PHASE1=10 \
EVAL_LIMIT=200 \
python experiments/paperstyle_rl/run_pipeline.py --stage phase1
```

## Notes

- Default embedding mode is `gnn_rag_gnn_exact`.
- In exact mode, query and passage prefixes are forced to empty.
- `DATASET=all` preprocess is not supported in `run_embed.sh`. Use `DATASET=cwq` or `DATASET=webqsp`.
- Windows wrappers are available at `experiments/paperstyle_rl/run_pipeline.ps1` and `experiments/paperstyle_rl/run_pipeline.cmd`.
- See `experiments/paperstyle_rl/README.md` for full pipeline details.
