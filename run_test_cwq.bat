@echo off

echo.
echo [1/1] Running Evaluation on Checkpoint 1 (Low VRAM Mode)...

python trm_unified\train_core.py ^
  --model_impl trm_hier6 ^
  --trm_root TinyRecursiveModels ^
  --ckpt checkpoints\model_ep10.pt ^
  --eval_json data\CWQ\test_split.jsonl ^

  --eval_batch_size 1 ^
  --entities_txt data\CWQ\entities.txt ^
  --relations_txt data\CWQ\relations.txt ^
  --query_emb_eval_npy trm_agent\emb\cwq_e5\query_dev.npy ^
  --entity_emb_npy trm_agent\emb\cwq_e5\entity_embeddings.npy ^
  --relation_emb_npy trm_agent\emb\cwq_e5\relation_embeddings.npy ^
  --eval_max_steps 3 ^
  --eval_max_neighbors 128 ^
  --eval_prune_keep 128 ^
  --eval_no_cycle ^
  --query_residual_enabled --query_residual_alpha 0.5 --query_residual_mode add ^
  --wandb_mode disabled --wandb_project graph-traverse
