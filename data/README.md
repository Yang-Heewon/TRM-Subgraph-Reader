# Data Setup (Google Drive)

아래 스크립트는 제공된 Google Drive 파일에서 데이터를 내려받고, 저장소 표준 경로로 자동 정리합니다.

- 기본 파일 URL:
  - `https://drive.google.com/file/d/13DduGb1C-O6udi744WxNVnOVDJ6122JG/view?usp=sharing`

## 1) 다운로드 + 경로 정리
```bash
bash scripts/download_data.sh
```

필수:
- `gdown` 설치 필요 (`pip install gdown`)

옵션:
- 다른 파일 URL 사용
```bash
GDRIVE_FILE_URL="https://drive.google.com/file/d/.../view?usp=sharing" bash scripts/download_data.sh
```
- 기존 폴더 URL을 계속 쓰고 싶다면
```bash
GDRIVE_FILE_URL="" GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/...." bash scripts/download_data.sh
```
- Google Drive 다운로드를 건너뛰고 직접 URL만 사용
```bash
SKIP_GDRIVE=1 WEBQSP_URL="https://..." CWQ_URL="https://..." bash scripts/download_data.sh
```

## 2) 전처리까지 한 번에
```bash
DATASET=webqsp bash scripts/setup_and_preprocess.sh
```

- `DATASET=cwq` 또는 `DATASET=all` 가능
- `DATASET=all`은 `CWQ -> WebQSP` 순서로 전처리합니다.
- BFS 깊이/경로 개수 조절은 `--override`로 가능:
```bash
python -m trm_rag_style.run --dataset cwq --stage preprocess --override max_steps=6 max_paths=8 mine_max_neighbors=256
```

## 기대 경로
WebQSP:
- `data/webqsp/train.json`
- `data/webqsp/dev.json`
- `data/webqsp/entities.txt`
- `data/webqsp/relations.txt`

CWQ:
- `data/CWQ/train_split.jsonl`
- `data/CWQ/dev_split.jsonl`
- `data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt`
- `data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt`
