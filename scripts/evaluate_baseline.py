"""
Baseline Evaluation Script

Evaluates the base Qwen-2.5-Coder model (without any fine-tuning)
to establish a baseline SSPR for comparison with the DPO-aligned model.

Usage:
    python scripts/evaluate_baseline.py --data_path data/synthetic_triplets.jsonl
"""
import logging
import argparse
from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.evaluation import run_evaluation, save_evaluation_artifacts
from src.model import load_model_and_tokenizer, prepare_model_for_inference
from src.utils import setup_logging

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--model_name", type=str, default=BASE_MODEL, help="Base model to evaluate")
    parser.add_argument("--output_dir", type=str, default="logs/eval_baseline", help="Directory for evaluation artifacts")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Evaluating baseline model: {args.model_name}")

    # Load Base Model (Inference Mode)
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    prepare_model_for_inference(model)
    summary, predictions = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_new_tokens=args.max_new_tokens,
    )
    artifact_paths = save_evaluation_artifacts(
        output_dir=args.output_dir,
        summary=summary,
        predictions=predictions,
        metadata={"model_name": args.model_name},
    )

    # Print detailed results
    print("\n" + "=" * 50)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Passed: {summary['passed']}")
    print(f"Valid JSON: {summary['valid_json']} ({summary['valid_json_rate']:.2%})")
    print(f"Exact Matches: {summary['exact_matches']} ({summary['exact_match_rate']:.2%})")
    print(f"SSPR (Strict Schema Pass Rate): {summary['strict_schema_pass_rate']:.2%}")
    print("\nFailure Breakdown:")
    for error_type, count in summary["failure_breakdown"].items():
        if count > 0:
            print(f"  - {error_type}: {count} ({count/summary['total_samples']*100:.1f}%)")
    print("\nArtifacts:")
    for label, path in artifact_paths.items():
        print(f"  - {label}: {path}")
    print("=" * 50)

    logging.info("Baseline SSPR: %.2f%%", summary["strict_schema_pass_rate"] * 100)

if __name__ == "__main__":
    main()
