"""
Baseline Evaluation Script

Evaluates the base Qwen-2.5-Coder model (without any fine-tuning)
to establish a baseline SSPR for comparison with the DPO-aligned model.

Usage:
    python scripts/evaluate_baseline.py --data_path data/synthetic_triplets.jsonl
"""
import json
import logging
import argparse
from tqdm import tqdm
from unsloth import FastLanguageModel
from src.validation import validate_tool_call
from src.utils import setup_logging, get_device

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = """You are a tool-calling assistant. When given a user request and tool specification, respond with ONLY a valid JSON object representing the tool call. Do not include any explanation, markdown formatting, or code blocks. Output raw JSON only."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--model_name", type=str, default=BASE_MODEL, help="Base model to evaluate")
    args = parser.parse_args()

    setup_logging()

    device = get_device()
    logging.info(f"Using device: {device}")
    logging.info(f"Evaluating baseline model: {args.model_name}")

    # Load Base Model (Inference Mode)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    passed = 0
    total = 0
    failures_by_type = {
        "json_error": 0,
        "hallucinated_param": 0,
        "type_mismatch": 0,
        "enum_violation": 0,
        "missing_required": 0,
        "other_schema_error": 0
    }

    logging.info(f"Starting baseline evaluation on {len(data)} samples...")

    for item in tqdm(data):
        # Manually constructing the prompt to match training format (with system prompt)
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode only the generated tokens
        response = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Validate against the SCHEMA saved in the dataset
        is_valid, error = validate_tool_call(response, item["schema"])

        if is_valid:
            passed += 1
        else:
            # Categorize the error
            if "Invalid JSON" in error:
                failures_by_type["json_error"] += 1
            elif "Additional properties" in error:
                failures_by_type["hallucinated_param"] += 1
            elif "is not of type" in error:
                failures_by_type["type_mismatch"] += 1
            elif "is not one of" in error:
                failures_by_type["enum_violation"] += 1
            elif "is a required property" in error:
                failures_by_type["missing_required"] += 1
            else:
                failures_by_type["other_schema_error"] += 1
            logging.debug(f"Failed: {error} | Response: {response}")

        total += 1

    score = passed / total if total > 0 else 0.0

    # Print detailed results
    print("\n" + "=" * 50)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Total Samples: {total}")
    print(f"Passed: {passed}")
    print(f"SSPR (Strict Schema Pass Rate): {score:.2%}")
    print("\nFailure Breakdown:")
    for error_type, count in failures_by_type.items():
        if count > 0:
            print(f"  - {error_type}: {count} ({count/total*100:.1f}%)")
    print("=" * 50)

    logging.info(f"Baseline SSPR: {score:.2%}")

if __name__ == "__main__":
    main()
