# TRM-Subgraph-Reader

이 레포는 **CWQ 기준 Subgraph Reader 실험 2개 모드만** 재현하도록 정리된 버전입니다.

- `v2`: BCE 기반 안정 학습(기본 실험)
- `hit1boost`: 방향 분리 + 랭킹 + hard negative 강화 실험

불필요한 RL/phase 스크립트는 제외했고, **데이터 다운로드부터 테스트까지** 한 흐름으로 실행할 수 있게 구성했습니다.

## 1) 구성

핵심 실행 파일:

- `trm_rag_style/scripts/run_download.sh`
- `trm_rag_style/scripts/run_embed.sh`
- `trm_rag_style/scripts/run_test.sh`
- `trm_rag_style/scripts/run_train_subgraph_v2_resume.sh`
- `trm_rag_style/scripts/run_train_subgraph_hit1boost.sh`
- `trm_rag_style/scripts/run_all_v2.sh`
- `trm_rag_style/scripts/run_all_hit1boost.sh`

핵심 코드:

- `trm_unified/train_core.py`
- `trm_unified/subgraph_reader.py`
- `trm_unified/embedder.py`
- `trm_rag_style/trm_pipeline/*.py`

## 2) 환경 준비

```bash
cd /path/to/TRM-Subgraph-Reader
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

권장:

- CUDA 환경에서 실행
- `wandb` 사용 시 `wandb login`

## 3) 데이터 다운로드 (CWQ)

```bash
cd /path/to/TRM-Subgraph-Reader
DATASET=cwq bash trm_rag_style/scripts/run_download.sh
```

생성 확인:

- `data/CWQ/train_split.jsonl`
- `data/CWQ/dev_split.jsonl`
- `data/CWQ/test_split.jsonl`
- `data/CWQ/entities.txt`
- `data/CWQ/relations.txt`

## 4) 전처리 + 임베딩

기본(E5):

```bash
cd /path/to/TRM-Subgraph-Reader
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
RUN_PREPROCESS=1 \
PREPROCESS_WORKERS=4 \
EMBED_GPUS=0,1,2,3 \
bash trm_rag_style/scripts/run_embed.sh
```

생성 확인:

- `trm_agent/processed/cwq/train.jsonl`
- `trm_agent/processed/cwq/dev.jsonl`
- `trm_agent/processed/cwq/test.jsonl`
- `trm_agent/emb/cwq_e5/entity_embeddings.npy`
- `trm_agent/emb/cwq_e5/relation_embeddings.npy`
- `trm_agent/emb/cwq_e5/query_train.npy`
- `trm_agent/emb/cwq_e5/query_dev.npy`
- `trm_agent/emb/cwq_e5/query_test.npy`

## 5) 학습 모드 A: v2

### 5-1. 처음부터 학습

```bash
cd /path/to/TRM-Subgraph-Reader
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
MASTER_PORT=29606 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMB_DIR=trm_agent/emb/cwq_e5 \
CKPT= \
SUBGRAPH_RESUME_EPOCH=-1 \
EPOCHS=50 \
BATCH_SIZE=1 \
EVAL_LIMIT=-1 \
CKPT_DIR=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2 \
WANDB_MODE=online \
WANDB_RUN_NAME=cwq_v2_scratch \
bash trm_rag_style/scripts/run_train_subgraph_v2_resume.sh
```

### 5-2. 이어서 학습(resume)

```bash
cd /path/to/TRM-Subgraph-Reader
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
MASTER_PORT=29606 \
CKPT=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2/model_ep30.pt \
SUBGRAPH_RESUME_EPOCH=30 \
EPOCHS=20 \
BATCH_SIZE=1 \
EVAL_LIMIT=-1 \
CKPT_DIR=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2 \
WANDB_MODE=online \
WANDB_RUN_NAME=cwq_v2_resume_ep30 \
bash trm_rag_style/scripts/run_train_subgraph_v2_resume.sh
```

## 6) 학습 모드 B: hit1boost

```bash
cd /path/to/TRM-Subgraph-Reader
CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
MASTER_PORT=29608 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMB_DIR=trm_agent/emb/cwq_e5 \
CKPT= \
SUBGRAPH_RESUME_EPOCH=-1 \
EPOCHS=50 \
BATCH_SIZE=1 \
EVAL_LIMIT=-1 \
SUBGRAPH_RECURSION_STEPS=12 \
SUBGRAPH_MAX_NODES=2048 \
SUBGRAPH_HOPS=3 \
SUBGRAPH_MAX_EDGES=8192 \
CKPT_DIR=trm_agent/ckpt/cwq_trm_hier6_subgraph_hit1boost_gpu123 \
WANDB_MODE=online \
WANDB_RUN_NAME=cwq_hit1boost_gpu123_scratch \
bash trm_rag_style/scripts/run_train_subgraph_hit1boost.sh
```

## 7) 단일 체크포인트 테스트

`run_test.sh`는 `CKPT` 경로에 `hit1boost` 문자열이 있으면 방향 분리 옵션을 자동으로 맞춥니다.

```bash
cd /path/to/TRM-Subgraph-Reader
CUDA_VISIBLE_DEVICES=0 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMB_DIR=trm_agent/emb/cwq_e5 \
CKPT=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2/model_ep27.pt \
EVAL_LIMIT=-1 \
BATCH_SIZE=8 \
SUBGRAPH_RECURSION_STEPS=12 \
SUBGRAPH_MAX_NODES=2048 \
SUBGRAPH_MAX_EDGES=8192 \
bash trm_rag_style/scripts/run_test.sh
```

## 8) epoch 20 이상 체크포인트 일괄 테스트

```bash
cd /path/to/TRM-Subgraph-Reader
mkdir -p logs

for d in \
  trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2 \
  trm_agent/ckpt/cwq_trm_hier6_subgraph_hit1boost_gpu123
 do
  for p in "$d"/model_ep*.pt; do
    [ -e "$p" ] || continue
    ep="$(basename "$p" | sed -E 's/model_ep([0-9]+)\.pt/\1/')"
    [ "$ep" -ge 20 ] || continue
    name="$(basename "$d")_ep${ep}"
    echo "[RUN] $name"
    CUDA_VISIBLE_DEVICES=0 \
    DATASET=cwq \
    EMB_MODEL=intfloat/multilingual-e5-large \
    EMB_TAG=e5 \
    EMB_DIR=trm_agent/emb/cwq_e5 \
    CKPT="$p" \
    EVAL_LIMIT=-1 \
    BATCH_SIZE=8 \
    SUBGRAPH_RECURSION_STEPS=12 \
    SUBGRAPH_MAX_NODES=2048 \
    SUBGRAPH_MAX_EDGES=8192 \
    bash trm_rag_style/scripts/run_test.sh \
      > "logs/test_${name}.log" 2>&1
  done
done

grep -nE "\[Test-Subgraph\]" logs/test_*.log
```

## 9) 올인원 실행

### v2

```bash
cd /path/to/TRM-Subgraph-Reader
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
MASTER_PORT=29606 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMBED_GPUS=0,1,2,3 \
PREPROCESS_WORKERS=4 \
bash trm_rag_style/scripts/run_all_v2.sh
```

### hit1boost

```bash
cd /path/to/TRM-Subgraph-Reader
CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
MASTER_PORT=29608 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMBED_GPUS=0,1,2,3 \
PREPROCESS_WORKERS=4 \
bash trm_rag_style/scripts/run_all_hit1boost.sh
```

## 10) 참고

- tqdm 진행바가 로그 리다이렉트 시 줄바꿈으로 보이게 되어 있습니다.
- 학습 후 자동 테스트는 `trm_rag_style/configs/base.json`의 `auto_test_after_train=true`로 동작합니다.
- 재현성 고정을 원하면 학습 실행 시 `seed`, `deterministic` override를 명시하세요.
