import sys
sys.path.append('.')

from trm_unified.data import preprocess_split

if __name__ == '__main__':
    print("Preprocessing train.json...")
    preprocess_split(
        dataset="webqsp",
        input_path="data/webqsp/train.json",
        output_path="data/webqsp/train.jsonl",
        entities_txt="data/webqsp/entities.txt",
        max_steps=3,
        max_paths=100,
        max_neighbors=256,
        mine_paths=True,
        require_valid_paths=True,
        preprocess_workers=1,
        progress_desc="preprocess_train"
    )

    print("Preprocessing dev.json...")
    preprocess_split(
        dataset="webqsp",
        input_path="data/webqsp/dev.json",
        output_path="data/webqsp/dev.jsonl",
        entities_txt="data/webqsp/entities.txt",
        max_steps=3,
        max_paths=100,
        max_neighbors=256,
        mine_paths=True,
        require_valid_paths=True,
        preprocess_workers=1,
        progress_desc="preprocess_dev"
    )
