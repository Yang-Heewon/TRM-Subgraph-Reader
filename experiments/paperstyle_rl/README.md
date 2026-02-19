# Paperstyle -> RL Preset

이 폴더는 다음 실험 흐름만 분리해서 실행하기 위한 전용 preset입니다.

1. `논문작업` 스타일 전처리/임베딩 재현
2. TRM phase1 학습 (relation CE 중심)
3. phase1 체크포인트에서 RL phase2 추가 파인튜닝
4. phase2 체크포인트로 test 평가(Hit@1/F1) 산출

## 빠른 실행(권장)

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
bash experiments/paperstyle_rl/run_all_wandb.sh
```

`run_all_wandb.sh`가 기본 진입점입니다. 내부에서 `run_all.sh`를 호출해
`00_preprocess -> 01_embed -> 02_train_phase1 -> 03_train_phase2_rl -> 04_eval_phase2_test`
순서로 끝까지 실행합니다.

## 단계별 실행

```bash
bash experiments/paperstyle_rl/00_preprocess.sh
bash experiments/paperstyle_rl/01_embed.sh
bash experiments/paperstyle_rl/02_train_phase1.sh
bash experiments/paperstyle_rl/03_train_phase2_rl.sh
bash experiments/paperstyle_rl/04_eval_phase2_test.sh
```

## 주요 환경변수

공통 변수는 `experiments/paperstyle_rl/env.sh`에 있습니다.

- `DATASET` (default: `cwq`)
- `EPOCHS_PHASE1` (default: `5`)
- `EPOCHS_PHASE2` (default: `20`)
- `BATCH_SIZE_PHASE1` (default: `6`)
- `BATCH_SIZE_PHASE2` (default: `2`)
- `LR` (default: `2e-4`)
- `WANDB_ENTITY` (default: `heewon6205-chung-ang-university`)
- `NPROC_PER_NODE` (phase1 default: `3`)
- `NPROC_PER_NODE_PHASE2` (phase2 default: `1`, RL 안정성용)
- `CKPT_DIR_PHASE1`, `CKPT_DIR_PHASE2`
- `EMBED_STYLE` (default: `gnn_rag_gnn_exact`)
- `EMBED_BACKEND` (default: `sentence_transformers`)
- `EMB_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBED_STYLE` 기본값은 `gnn_rag_gnn_exact`이며,
  relation 텍스트를 GNN-RAG `gnn`과 동일하게 `fields[-2] + fields[-1]` 규칙으로 만듭니다.
- `TRAIN_STYLE` (default: `gnn_rag`, phase1을 relation CE 중심으로 고정)
- `ENTITY_NAMES_JSON` (default: `data/data/entities_names.json`)
- `RESULTS_DIR` (default: `experiments/paperstyle_rl/results`)

예시:

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
EPOCHS_PHASE1=20 EPOCHS_PHASE2=10 WANDB_ENTITY=<your_entity> \
bash experiments/paperstyle_rl/run_all.sh
```

## 참고

- phase2 스크립트는 기본으로 phase1의 마지막 체크포인트를 자동 탐색합니다.
- 자동 탐색이 안되면 `PHASE1_CKPT=/path/to/model_epX.pt`를 지정하세요.
- 최종 test 지표는 `experiments/paperstyle_rl/results/*.summary.txt`에 저장됩니다.
