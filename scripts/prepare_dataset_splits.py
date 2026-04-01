import argparse
import json
import os

from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.dataset_split import prepare_dataset_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="Source JSONL with synthetic triplets")
    parser.add_argument("--train_path", type=str, default="data/train_triplets.jsonl")
    parser.add_argument("--val_path", type=str, default="data/val_triplets.jsonl")
    parser.add_argument("--test_path", type=str, default="data/test_triplets.jsonl")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedupe_by", type=str, default="prompt", choices=["prompt", "pair", "none"])
    parser.add_argument("--report_path", type=str, default="logs/dataset_split_summary.json")
    args = parser.parse_args()

    report = prepare_dataset_splits(
        source_path=args.source_path,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dedupe_by=args.dedupe_by,
    )

    if args.report_path:
        parent = os.path.dirname(args.report_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.report_path, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    print("=" * 50)
    print("DATASET SPLITS")
    print("=" * 50)
    print(f"Source Rows: {report['source_rows']}")
    print(f"Valid Rows: {report['valid_rows']}")
    print(f"Deduped Rows: {report['deduped_rows']}")
    print(f"Train Rows: {report['train_rows']} -> {report['train_path']}")
    print(f"Val Rows: {report['val_rows']} -> {report['val_path']}")
    print(f"Test Rows: {report['test_rows']} -> {report['test_path']}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
