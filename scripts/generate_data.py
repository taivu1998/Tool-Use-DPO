import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

from _bootstrap import PROJECT_ROOT  # noqa: F401
from dotenv import load_dotenv
from openai import OpenAI

from src.utils import setup_logging
from src.validation import VALIDATION_MODE_STRICT, validate_tool_call_detailed

load_dotenv()

SYSTEM_PROMPT = """
You are a Synthetic Data Generator for an LLM Alignment project.
Your goal is to generate "Hard Negative" DPO triplets for Tool Use.

Triplets: (Prompt, Chosen, Rejected)

1. Prompt: A user query requiring a specific tool call.
2. Chosen: A PERFECTLY valid JSON tool call adhering to the schema.
3. Rejected: A SUBTLY incorrect tool call. It MUST be valid JSON, but it must fail the schema in one of these specific ways:
   - Hallucinated Parameter: Add a plausible argument not in the schema.
   - Type Mismatch: Pass a string "5" where an integer 5 is required.
   - Enum Violation: Pass "urgent" when allowed values are ["high", "medium", "low"].
   - Missing Required: Omit a mandatory argument.

Requirements:
- Return an object with a top-level "examples" array.
- Each example must contain keys: "prompt", "chosen", "rejected", "schema".
- The "schema" must be a full JSON schema for the chosen tool call.
- Object schemas should be strict unless the structure is intentionally open-ended.
"""


def generate_batch(client: OpenAI, model: str, num_samples: int) -> List[Dict[str, Any]]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate {num_samples} unique, diverse tool-use triplets."},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    payload = json.loads(content)
    examples = payload.get("examples", payload.get("triplets"))
    if not isinstance(examples, list):
        raise ValueError("Model response did not include an 'examples' list.")
    return examples


def validate_generated_example(item: Dict[str, Any]) -> Dict[str, Any]:
    required_fields = {"prompt", "chosen", "rejected", "schema"}
    missing_fields = sorted(required_fields - set(item))
    if missing_fields:
        raise ValueError(f"Generated example is missing required fields: {missing_fields}")

    schema = item["schema"]
    chosen_str = item["chosen"] if isinstance(item["chosen"], str) else json.dumps(item["chosen"])
    rejected_str = item["rejected"] if isinstance(item["rejected"], str) else json.dumps(item["rejected"])

    chosen_result = validate_tool_call_detailed(chosen_str, schema, mode=VALIDATION_MODE_STRICT)
    rejected_result = validate_tool_call_detailed(rejected_str, schema, mode=VALIDATION_MODE_STRICT)

    return {
        "chosen_result": chosen_result,
        "rejected_result": rejected_result,
        "chosen_str": chosen_str,
        "rejected_str": rejected_str,
    }


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="data/synthetic_triplets.jsonl")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max_consecutive_failures", type=int, default=5)
    parser.add_argument("--retry_delay_seconds", type=float, default=2.0)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running data generation.")

    client = OpenAI(api_key=api_key)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stats = {
        "accepted": 0,
        "discarded_invalid_chosen": 0,
        "discarded_valid_rejected": 0,
        "discarded_bad_schema": 0,
        "api_failures": 0,
    }
    consecutive_failures = 0

    with open(args.output_file, "w") as handle:
        while stats["accepted"] < args.num_samples:
            remaining = args.num_samples - stats["accepted"]
            batch_target = min(args.batch_size, remaining)
            logging.info("Generating batch... (%s/%s)", stats["accepted"], args.num_samples)

            try:
                batch = generate_batch(client, model=args.model, num_samples=batch_target)
            except Exception as exc:
                stats["api_failures"] += 1
                consecutive_failures += 1
                logging.error("API/generation failure: %s", exc)
                if consecutive_failures >= args.max_consecutive_failures:
                    raise SystemExit(
                        f"Aborting after {consecutive_failures} consecutive generation failures."
                    ) from exc
                time.sleep(args.retry_delay_seconds)
                continue

            batch_accepted = 0
            for item in batch:
                try:
                    validated = validate_generated_example(item)
                except Exception as exc:
                    stats["discarded_bad_schema"] += 1
                    logging.warning("Discarded sample due to malformed payload or schema: %s", exc)
                    continue

                chosen_result = validated["chosen_result"]
                rejected_result = validated["rejected_result"]

                if not chosen_result.is_valid:
                    stats["discarded_invalid_chosen"] += 1
                    logging.warning("Discarded sample: chosen output invalid (%s)", chosen_result.message)
                    continue

                if rejected_result.is_valid:
                    stats["discarded_valid_rejected"] += 1
                    logging.warning("Discarded sample: rejected output unexpectedly valid.")
                    continue

                item["schema"] = chosen_result.normalized_schema
                item["chosen"] = chosen_result.normalized_json or validated["chosen_str"]
                item["rejected"] = rejected_result.normalized_json or validated["rejected_str"]
                handle.write(json.dumps(item, sort_keys=True) + "\n")
                stats["accepted"] += 1
                batch_accepted += 1

                if stats["accepted"] >= args.num_samples:
                    break

            if batch_accepted == 0:
                consecutive_failures += 1
                logging.warning("No valid samples accepted from the last batch.")
                if consecutive_failures >= args.max_consecutive_failures:
                    raise SystemExit(
                        f"Aborting after {consecutive_failures} consecutive empty/failed batches."
                    )
                time.sleep(args.retry_delay_seconds)
            else:
                consecutive_failures = 0

    logging.info("Generation completed with stats: %s", stats)


if __name__ == "__main__":
    main()
