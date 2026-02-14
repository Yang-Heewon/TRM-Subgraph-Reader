import json
import os


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_example(idx, q, start_entity, answer_entity, relation_path):
    tuples = []
    cur = start_entity
    for rel_id, nxt in relation_path:
        tuples.append([cur, rel_id, nxt])
        cur = nxt
    # add some distractor edges
    tuples.append([start_entity, 2, 4])
    tuples.append([4, 1, 5])
    return {
        "id": f"demo-{idx}",
        "question": q,
        "entities": [start_entity],
        "answers": [{"kb_id": f"m.e{answer_entity}", "text": f"Entity {answer_entity}"}],
        "subgraph": {"tuples": tuples},
    }


def main():
    out_dir = os.path.join("data", "webqsp")
    entities = [
        "m.e0\tEntity 0",
        "m.e1\tEntity 1",
        "m.e2\tEntity 2",
        "m.e3\tEntity 3",
        "m.e4\tEntity 4",
        "m.e5\tEntity 5",
    ]
    relations = [
        "rel_to_1",
        "rel_to_2",
        "rel_noise",
    ]

    train = [
        build_example(
            0,
            "Which entity is reached by going to one then two from entity zero?",
            start_entity=0,
            answer_entity=2,
            relation_path=[(0, 1), (1, 2)],
        ),
        build_example(
            1,
            "From entity three, which node do we reach via relation zero?",
            start_entity=3,
            answer_entity=1,
            relation_path=[(0, 1)],
        ),
    ]

    dev = [
        build_example(
            2,
            "Starting at zero and following relation zero then one, where do we end?",
            start_entity=0,
            answer_entity=2,
            relation_path=[(0, 1), (1, 2)],
        ),
    ]

    write_lines(os.path.join(out_dir, "entities.txt"), entities)
    write_lines(os.path.join(out_dir, "relations.txt"), relations)
    write_json(os.path.join(out_dir, "train.json"), train)
    write_json(os.path.join(out_dir, "dev.json"), dev)
    print(f"✅ demo WebQSP data written to {out_dir}")


if __name__ == "__main__":
    main()
