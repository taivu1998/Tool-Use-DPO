import json
import os
import argparse
import logging
from dotenv import load_dotenv
from openai import OpenAI
from src.utils import setup_logging
from src.validation import validate_tool_call

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """
You are a Synthetic Data Generator for an LLM Alignment project.
Your goal is to generate "Hard Negative" DPO triplets for Tool Use.

Triplets: (Prompt, Chosen, Rejected)

1. **Prompt**: A user query requiring a specific tool call.
2. **Chosen**: A PERFECTLY valid JSON tool call adhering to the schema.
3. **Rejected**: A SUBTLY incorrect tool call. It MUST be valid JSON, but it must fail the schema in one of these specific ways:
   - **Hallucinated Parameter**: Add a plausible argument not in the schema.
   - **Type Mismatch**: Pass a string "5" where an integer 5 is required.
   - **Enum Violation**: Pass "urgent" when allowed values are ["high", "medium", "low"].
   - **Missing Required**: Omit a mandatory argument.

Output Format: JSON Lines. Each line must contain keys: "prompt", "chosen", "rejected", "schema".
The "schema" key should contain the full JSON schema used for validation.
"""

def generate_batch(client, num_samples=10):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate {num_samples} unique, diverse tool-use triplets."}
        ],
        response_format={"type": "json_object"}
    )
    # GPT-4o usually returns a dictionary with a key like "triplets" or "examples"
    content = completion.choices[0].message.content
    data = json.loads(content)
    # Flexible parsing
    return data.get("examples", data.get("triplets", [data]))

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="data/synthetic_triplets.jsonl")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    valid_samples = 0
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        while valid_samples < args.num_samples:
            try:
                logging.info(f"Generating batch... ({valid_samples}/{args.num_samples})")
                batch = generate_batch(client, num_samples=min(5, args.num_samples - valid_samples)) 
                
                for item in batch:
                    # STRICT VALIDATION GATE
                    schema = item.get("schema")

                    # Ensure chosen and rejected are JSON strings for validation
                    chosen_str = item["chosen"] if isinstance(item["chosen"], str) else json.dumps(item["chosen"])
                    rejected_str = item["rejected"] if isinstance(item["rejected"], str) else json.dumps(item["rejected"])

                    chosen_valid, _ = validate_tool_call(chosen_str, schema)
                    rejected_valid, _ = validate_tool_call(rejected_str, schema)

                    if chosen_valid and not rejected_valid:
                        # Store as strings for consistency with training format
                        item["chosen"] = chosen_str
                        item["rejected"] = rejected_str
                        f.write(json.dumps(item) + "\n")
                        valid_samples += 1
                    else:
                        logging.warning("Discarded sample: Validation logic failed (Chosen invalid or Rejected valid).")
                        
            except Exception as e:
                logging.error(f"Error during generation: {e}")

if __name__ == "__main__":
    main()