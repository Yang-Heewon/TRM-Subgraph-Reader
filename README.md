# GRAPH-TRAVERSE (TRM Subgraph Reader)

This repository contains the refactored **Tiny Recursive Model (TRM)**, now operating as a **Subgraph Reader**. 

## What's New: Subgraph Reader Architecture
The original TRM model operated sequentially, moving from node to node via Reinforcement Learning and Beam Search. This repository implements a full **Subgraph Reader Overhaul**:
- **Global Graph Processing**: The model now ingests the entire N-hop subgraph simultaneously.
- **Node Classification**: Instead of a "halt-or-continue" sequential head, it features a direct `node_cls_head` predicting Binary Cross-Entropy (BCE) over all $N$ subgraph nodes in a single forward pass.
- **Batched Attention Modeling**: A custom Adjacency Matrix (`am`) provides structural attention masks, enforcing that node tokens only attend to connected neighbors or the central textual question tokens.
- **Batched High-Speed Evaluation**: Evaluation is no longer restricted to single-item sequential decoding. The `evaluate_relation_beam` has been rewritten to support batched evaluation (`eval_batch_size=100`), dramatically accelerating testing.

---

## Main Entrypoints

- `trm_agent/run.py` (Main Python Pipeline)
- `trm_rag_style/scripts/run_download.sh`
- `trm_rag_style/scripts/run_embed.sh`
- `experiments/paperstyle_rl/run_pipeline.py`

## Download & Prep Data

Full CWQ from HF:
```bash
python experiments/paperstyle_rl/run_pipeline.py --stage download
# Or via script
bash scripts/download_cwq_hf.sh
```

Embed Graph Data:
```bash
DATASET=webqsp \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
bash trm_rag_style/scripts/run_embed.sh
```

## Training the Subgraph Reader

**Warning:** In Windows environments, executing distributed training using PyTorch's `torchrun` and `nccl` backends may crash due to missing `libuv`. We recommend executing the standalone Python command or modifying your DDP timeout settings to use `gloo`.

To execute a full native training run on a single machine/GPU without `torchrun` crashes on Windows:

```bash
python -m trm_agent.run --dataset webqsp --model_impl trm_hier6 --stage train --override epochs=20 batch_size=8 lr=1e-4 eval_limit=-1 eval_batch_size=100
```
- Set `eval_batch_size=100` to utilize the new rapid Subgraph batched forward pass logic.
- Change `--stage train` to `--stage all` if you have yet to preprocess and embed your local dataset.

## High-Speed Batched Evaluation

If you have a trained checkpoint and want to test it against the complete Dev or Test split rapidly:

```bash
python -m trm_agent.run \
  --dataset webqsp \
  --model_impl trm_hier6 \
  --stage test \
  --ckpt "path_to_model_epX.pt" \
  --override \
    eval_limit=-1 \
    eval_batch_size=100
```
