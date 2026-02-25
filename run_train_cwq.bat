@echo off

echo.
echo [1/1] Resuming Training from Epoch 10 with W&B...
set USE_LIBUV=0
set GLOO_SOCKET_IFNAME=127.0.0.1
set DDP_TIMEOUT_MINUTES=120



python trm_unified\train_core.py ^
  --model_impl trm_hier6 ^
  --trm_root TinyRecursiveModels ^
  --max_steps 3 ^
  --max_paths 3 ^
  --max_neighbors 128 ^
  --pos_encodings 0 ^
  --ckpt_name trm_cwq_subgraph_reader ^
  --train_json data\CWQ\train.jsonl ^
  --dev_json data\CWQ\dev.jsonl ^
  --eval_limit 500 --eval_batch_size 2 ^
  --entities_txt data\CWQ\entities.txt ^
  --relations_txt data\CWQ\relations.txt ^
  --q_npy trm_agent\emb\cwq_e5\query_train.npy ^
  --query_emb_dev_npy trm_agent\emb\cwq_e5\query_dev.npy ^
  --entity_emb_npy trm_agent\emb\cwq_e5\entity_embeddings.npy ^
  --relation_emb_npy trm_agent\emb\cwq_e5\relation_embeddings.npy ^
  --rel_npy trm_agent\emb\cwq_e5\relation_embeddings.npy ^
  --lr 1e-4 --batch_size 2 --epochs 50 ^
  --ckpt checkpoints\model_ep10.pt ^
  --eval_start 1 ^
  --eval_every 1 ^
  --prune_keep 256 --eval_prune_keep 256 ^
  --prune_rand 0 --eval_no_cycle ^
  --query_residual_enabled --query_residual_alpha 0.5 --query_residual_mode add ^
  --wandb_mode online --wandb_project antigravity

