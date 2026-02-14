#!/usr/bin/env python
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Build entity text file: <entity_id>\\t<name>")
    ap.add_argument("--entities_txt", required=True)
    ap.add_argument("--entity_names_json", required=True)
    ap.add_argument("--output_txt", required=True)
    args = ap.parse_args()

    with open(args.entity_names_json, "r", encoding="utf-8") as f:
        name_map = json.load(f)
    if not isinstance(name_map, dict):
        raise ValueError("entity_names_json must be a JSON object: {entity_id: name}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_txt)), exist_ok=True)
    written = 0
    with open(args.entities_txt, "r", encoding="utf-8") as fin, open(args.output_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            eid = line.strip().split("\t")[0]
            if not eid:
                continue
            name = name_map.get(eid, eid)
            fout.write(f"{eid}\t{name}\n")
            written += 1

    print(f"✅ wrote {written} lines to {args.output_txt}")


if __name__ == "__main__":
    main()
