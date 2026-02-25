import sys
sys.path.append('.')

from trm_unified.data import preprocess_split

if __name__ == '__main__':
    print("Preprocessing CWQ train_split.jsonl...")
    preprocess_split(
        dataset="cwq",
        input_path="data/CWQ/train_split.jsonl",
        output_path="data/CWQ/train.jsonl",
        entities_txt="data/CWQ/entities.txt",
        max_steps=3,
        max_paths=100,
        max_neighbors=256,
        mine_paths=True,
        require_valid_paths=True,
        preprocess_workers=1,
        progress_desc="preprocess_train"
    )

    print("Preprocessing CWQ dev_split.jsonl...")
    preprocess_split(
        dataset="cwq",
        input_path="data/CWQ/dev_split.jsonl",
        output_path="data/CWQ/dev.jsonl",
        entities_txt="data/CWQ/entities.txt",
        max_steps=3,
        max_paths=100,
        max_neighbors=256,
        mine_paths=True,
        require_valid_paths=True,
        preprocess_workers=1,
        progress_desc="preprocess_dev"
    )
