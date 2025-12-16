"""Debug script to examine actual model outputs"""
import json
import argparse
from unsloth import FastLanguageModel

SYSTEM_PROMPT = """You are a tool-calling assistant. When given a user request and tool specification, respond with ONLY a valid JSON object representing the tool call. Do not include any explanation, markdown formatting, or code blocks. Output raw JSON only."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/synthetic_triplets.jsonl")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Load data
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"Examining {args.num_samples} samples from {args.model_path}")
    print(f"{'='*60}\n")

    for i, item in enumerate(data[:args.num_samples]):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            do_sample=False,
        )

        response = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        print(f"--- Sample {i+1} ---")
        print(f"PROMPT: {item['prompt'][:100]}...")
        print(f"EXPECTED: {item['chosen'][:150]}...")
        print(f"ACTUAL OUTPUT: [{response}]")

        # Check if valid JSON
        try:
            json.loads(response)
            print("STATUS: ✓ Valid JSON")
        except json.JSONDecodeError as e:
            print(f"STATUS: ✗ Invalid JSON - {e}")
        print()

if __name__ == "__main__":
    main()
