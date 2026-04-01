import argparse
import sys

from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.dataset_audit import audit_dataset, write_audit_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSONL")
    parser.add_argument("--report_path", type=str, default=None, help="Optional path for a JSON audit report")
    parser.add_argument(
        "--fail_on_errors",
        action="store_true",
        help="Exit non-zero when invalid pairs are detected.",
    )
    parser.add_argument(
        "--fail_on_open_objects",
        action="store_true",
        help="Exit non-zero when any object schemas remain open-ended.",
    )
    parser.add_argument(
        "--fail_on_duplicate_prompts",
        action="store_true",
        help="Exit non-zero when duplicate prompts are detected.",
    )
    parser.add_argument(
        "--fail_on_duplicate_pairs",
        action="store_true",
        help="Exit non-zero when duplicate prompt/chosen/rejected rows are detected.",
    )
    args = parser.parse_args()

    report = audit_dataset(args.data_path)
    write_audit_report(report, args.report_path)

    print("=" * 50)
    print("DATASET AUDIT")
    print("=" * 50)
    print(f"Data Path: {report['data_path']}")
    print(f"Total Samples: {report['total_samples']}")
    print(f"Valid Pairs: {report['valid_pairs']}")
    print(f"Invalid Pairs: {report['invalid_pairs']}")
    print(f"Duplicate Prompts: {report['duplicate_prompt_count']}")
    print(f"Duplicate Pairs: {report['duplicate_pair_count']}")
    print(
        "Open Object Nodes: "
        f"{report['open_object_nodes']}/{report['total_object_nodes']}"
    )
    print("Error Counts:")
    if report["error_counts"]:
        for key, value in sorted(report["error_counts"].items()):
            print(f"  - {key}: {value}")
    else:
        print("  - none")
    print("=" * 50)

    if args.fail_on_errors and report["invalid_pairs"] > 0:
        return 1
    if args.fail_on_open_objects and report["open_object_nodes"] > 0:
        return 1
    if args.fail_on_duplicate_prompts and report["duplicate_prompt_count"] > 0:
        return 1
    if args.fail_on_duplicate_pairs and report["duplicate_pair_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
