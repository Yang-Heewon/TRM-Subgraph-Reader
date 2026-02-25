# GRAPH-TRAVERSE (TRM Subgraph Reader)

This repository implements the **Tiny Recursive Model (TRM)** optimized for complex subgraph reasoning over Knowledge Graphs, specifically tuned for the **CWQ (Complex WebQuestions)** dataset and fully configured for **native execution on Windows**.

## Architecture Highlights
- **Hop-3 Subgraph Input**: Takes an entire semantic subgraph surrounding the query entity at once contextually.
- **Deep Recursion (`L_cycles`, `H_cycles`)**: Recursively passes messages across graph nodes to find multi-hop answers.
- **Batched Attention & High-Speed Evaluation**: Customized sparse attention mechanisms handle 100+ node graphs rapidly without exploding VRAM, and evaluation paths are batched for speedy testing.

---

## 🚀 Quick Start: End-to-End Pipeline (Windows)

To execute the entire lifecycle (Preprocessing -> Embedding Extraction -> Training) in a single run:

```bat
run_pipeline_cwq.bat
```
*(This script will process the raw dataset, generate dense E5 vectors for text/relations, and immediately launch the deep training loop.)*

---

## 🛠 Step-by-Step Usage (Manual Execution)

If you prefer to run the stages iteratively to monitor outputs or adjust hyperparameters:

### 1. Data Preparation & Preprocessing
Extracts the raw CWQ dataset and clips a localized 3-hop subgraph around each question's subject entity.
```cmd
python preprocess_cwq.py
```
*Output:* `data/CWQ/train.jsonl`, `data/CWQ/dev.jsonl`

### 2. Embedding Extraction
Converts all textual queries, entity names, and relation labels into rich mathematical vector embeddings using the `intfloat/multilingual-e5-large` embedding model.
```cmd
python -m trm_agent.run --dataset cwq --stage embed --override embed_batch_size=256
```
*Output:* Generated `.npy` arrays inside `trm_agent/emb/cwq_e5/`.

### 3. Model Training
Initiates the TRM graph training loop. The current stable configuration uses `--hidden_size 512`, `--num_heads 8`, and `--max_paths 3` for high efficiency and stable Hit@1 scaling on 24GB VRAM.
```cmd
run_train_cwq.bat
```
*Note: Model weights will be saved periodically inside the `checkpoints/` directory. (e.g., `model_ep10.pt`). You can enable Weights & Biases (wandb) logging inside the script by changing `disabled` to `online`.*

### 4. Evaluation / Testing
To test the trained model on an unseen development or test split using the saved weights:
```cmd
run_test_cwq.bat
```
*If you wish to test a specific checkpoint, modify the `--ckpt checkpoints\model_epX.pt` argument inside the batch file before running.*
