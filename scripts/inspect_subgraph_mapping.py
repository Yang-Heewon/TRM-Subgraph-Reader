#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == "[":
            for ex in json.load(f):
                yield ex
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_entities(entities_txt: str) -> List[str]:
    out = []
    with open(entities_txt, "r", encoding="utf-8") as f:
        for line in f:
            eid = line.rstrip("\n").split("\t")[0].strip()
            out.append(eid)
    return out


def load_relations(relations_txt: str) -> List[str]:
    out = []
    with open(relations_txt, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            out.append(rel)
    return out


def load_names(entity_names_json: str) -> Dict[str, str]:
    if not entity_names_json:
        return {}
    with open(entity_names_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def safe_get(arr: List[str], idx):
    try:
        i = int(idx)
    except Exception:
        return str(idx), "<?>"
    if i < 0 or i >= len(arr):
        return str(idx), "<out-of-range>"
    return str(i), arr[i]


def main():
    ap = argparse.ArgumentParser(description="Inspect tuple-id to entity/relation/name mapping.")
    ap.add_argument("--input", required=True, help="train/dev/test json or jsonl")
    ap.add_argument("--entities_txt", required=True)
    ap.add_argument("--relations_txt", required=True)
    ap.add_argument("--entity_names_json", default="")
    ap.add_argument("--index", type=int, default=0, help="record index")
    ap.add_argument("--show_tuples", type=int, default=20, help="number of tuples to print")
    args = ap.parse_args()

    ents = load_entities(args.entities_txt)
    rels = load_relations(args.relations_txt)
    names = load_names(args.entity_names_json)

    rec = None
    for i, ex in enumerate(iter_records(args.input)):
        if i == args.index:
            rec = ex
            break
    if rec is None:
        raise IndexError(f"record index out of range: {args.index}")

    print(f"[record] index={args.index}")
    print("[question]", rec.get("question", ""))
    print("[start_entities]", rec.get("entities", []))
    print("[answers]", rec.get("answers", []))

    tuples = rec.get("subgraph", {}).get("tuples", [])
    print(f"[subgraph] tuples={len(tuples)} show={min(len(tuples), args.show_tuples)}")
    for t in tuples[: args.show_tuples]:
        if not isinstance(t, list) or len(t) != 3:
            continue
        s_idx, r_idx, o_idx = t
        _, s_key = safe_get(ents, s_idx)
        _, o_key = safe_get(ents, o_idx)
        _, rel = safe_get(rels, r_idx)
        s_name = names.get(s_key, s_key)
        o_name = names.get(o_key, o_key)
        print(f"  [{s_idx}, {r_idx}, {o_idx}] :: ({s_key}:{s_name}) -[{rel}]-> ({o_key}:{o_name})")


if __name__ == "__main__":
    main()
