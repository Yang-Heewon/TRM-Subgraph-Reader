# GRAPH-TRAVERSE (TRM Subgraph Reader)

This repository contains the refactored **Tiny Recursive Model (TRM)**, now operating as a **Subgraph Reader**. 

## What's New: Subgraph Reader Architecture
The original TRM model operated sequentially, moving from node to node via Reinforcement Learning and Beam Search. This repository implements a full **Subgraph Reader Overhaul**:
- **Global Graph Processing**: The model now ingests the entire N-hop subgraph simultaneously.
- **Node Classification**: Instead of a "halt-or-continue" sequential head, it features a direct `node_cls_head` predicting Binary Cross-Entropy (BCE) over all $N$ subgraph nodes in a single forward pass.
- **Batched Attention Modeling**: A custom Adjacency Matrix (`am`) provides structural attention masks, enforcing that node tokens only attend to connected neighbors or the central textual question tokens.
- **Batched High-Speed Evaluation**: Evaluation is no longer restricted to single-item sequential decoding. The `evaluate_relation_beam` has been rewritten to support batched evaluation (`eval_batch_size=100`), dramatically accelerating testing.

---

## 🚀 One-Command Full Pipeline

To run the complete process from generating data to training and testing in one go:
```bash
python -m trm_agent.run --dataset webqsp --model_impl trm_hier6 --stage all --override epochs=20 batch_size=8 lr=1e-4 eval_limit=-1 eval_batch_size=100
```
*Note: Run on a single line.*

---

## 🛠 Step-by-Step Usage

### 1. Data Download
Download the dataset locally (example: CWQ from HF):
```bash
bash scripts/download_cwq_hf.sh
```

### 2. Preprocessing
Parse the dataset, create maps, and initialize subgraphs:
```bash
python -m trm_agent.run --dataset webqsp --stage preprocess
```

### 3. Embedding (Model Customizable)
Vectorize your Nodes and Relations. You can swap out the embedding model according to your requirements:
```bash
DATASET=webqsp \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
bash trm_rag_style/scripts/run_embed.sh
```
Or directly from the entry point:
```bash
python -m trm_agent.run --dataset webqsp --stage embed --embedding_model intfloat/multilingual-e5-large
```

### 4. Training the Subgraph Reader
Execute a full native training run on a single machine without `torchrun` crashes on Windows:
```bash
python -m trm_agent.run --dataset webqsp --model_impl trm_hier6 --stage train --override epochs=20 batch_size=8 lr=1e-4 eval_limit=-1 eval_batch_size=100
```

### 5. High-Speed Batched Testing
Verify the accuracy (Hit@1, F1) on the test split:
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
