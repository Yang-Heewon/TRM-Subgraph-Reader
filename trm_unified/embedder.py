import json
import os
import hashlib
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from .data import iter_json_records


def read_text_lines(path: str, mode: str) -> List[str]:
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if mode == 'entity':
                if len(parts) >= 2 and parts[1].strip():
                    texts.append(parts[1].strip())
                else:
                    texts.append(parts[0].strip())
            else:
                texts.append(parts[0].strip())
    return texts


def collect_questions(jsonl_path: str) -> List[str]:
    qs = []
    for ex in iter_json_records(jsonl_path):
        q = ex.get('question', '')
        qs.append(q if q else '')
    return qs


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d


def _hash_vec(text: str, dim: int) -> np.ndarray:
    toks = (text or "").strip().split()
    if not toks:
        toks = ["<empty>"]
    v = np.zeros((dim,), dtype=np.float32)
    for t in toks:
        h = hashlib.sha1(t.encode("utf-8")).digest()
        for i in range(0, len(h), 2):
            slot = ((h[i] << 8) + h[i + 1]) % dim
            sign = 1.0 if (h[i] & 1) else -1.0
            v[slot] += sign
    n = np.linalg.norm(v) + 1e-12
    return v / n


def encode_texts_local_hash(texts: List[str], dim: int = 256, desc: str = "embed(local-hash)") -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack([_hash_vec(x, dim) for x in tqdm(texts, desc=desc, unit='txt')]).astype(np.float32)


def encode_texts(
    model_name: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str,
    embed_gpus: str = "",
    desc: str = "embed",
) -> np.ndarray:
    if (model_name or "").strip().lower() in {"local-hash", "local-simple", "local"}:
        return encode_texts_local_hash(texts, desc=f"{desc}(local-hash)")

    gpu_ids = _parse_gpu_ids(embed_gpus)
    run_device = device
    if torch.cuda.is_available() and gpu_ids:
        run_device = f"cuda:{gpu_ids[0]}"

    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(run_device)
        if torch.cuda.is_available() and len(gpu_ids) > 1:
            mdl = torch.nn.DataParallel(mdl, device_ids=gpu_ids, output_device=gpu_ids[0])
    except Exception:
        print(f"⚠️ failed to load embedding model '{model_name}', falling back to local-hash embeddings")
        return encode_texts_local_hash(texts, desc=f"{desc}(fallback)")
    mdl.eval()

    out = []
    total_batches = (len(texts) + batch_size - 1) // batch_size if batch_size > 0 else 0
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc=desc, unit='batch'):
            chunk = texts[i:i + batch_size]
            t = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            t = {k: v.to(run_device, non_blocking=True) for k, v in t.items()}
            out_obj = mdl(**t, return_dict=False)
            h = out_obj[0] if isinstance(out_obj, (tuple, list)) else out_obj.last_hidden_state
            emb = mean_pool(h, t['attention_mask'])
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def build_embeddings(
    model_name: str,
    entities_txt: str,
    relations_txt: str,
    train_jsonl: str,
    dev_jsonl: str,
    out_dir: str,
    batch_size: int = 64,
    max_length: int = 128,
    device: str = 'cuda',
    embed_gpus: str = "",
):
    os.makedirs(out_dir, exist_ok=True)

    ent_texts = read_text_lines(entities_txt, mode='entity')
    rel_texts = read_text_lines(relations_txt, mode='relation')
    q_train = collect_questions(train_jsonl)
    q_dev = collect_questions(dev_jsonl)

    ent = encode_texts(model_name, ent_texts, batch_size, max_length, device, embed_gpus=embed_gpus, desc='embed:entity')
    rel = encode_texts(model_name, rel_texts, batch_size, max_length, device, embed_gpus=embed_gpus, desc='embed:relation')
    qtr = encode_texts(model_name, q_train, batch_size, max_length, device, embed_gpus=embed_gpus, desc='embed:query_train')
    qdv = encode_texts(model_name, q_dev, batch_size, max_length, device, embed_gpus=embed_gpus, desc='embed:query_dev')

    np.save(os.path.join(out_dir, 'entity_embeddings.npy'), ent)
    np.save(os.path.join(out_dir, 'relation_embeddings.npy'), rel)
    np.save(os.path.join(out_dir, 'query_train.npy'), qtr)
    np.save(os.path.join(out_dir, 'query_dev.npy'), qdv)

    meta = {
        'model_name': model_name,
        'entity_shape': list(ent.shape),
        'relation_shape': list(rel.shape),
        'query_train_shape': list(qtr.shape),
        'query_dev_shape': list(qdv.shape),
    }
    with open(os.path.join(out_dir, 'embedding_meta.json'), 'w', encoding='utf-8') as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)
    return meta


def _parse_gpu_ids(embed_gpus: str) -> List[int]:
    raw = (embed_gpus or "").strip()
    if not raw:
        return []
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out
