"""Debug script to examine actual model outputs"""
import json
import argparse
from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.model import load_model_and_tokenizer, prepare_model_for_inference
from src.prompts import build_chat_prompt
from src.validation import validate_tool_call_detailed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/synthetic_triplets.jsonl")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    prepare_model_for_inference(model)

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
        prompt = build_chat_prompt(item["prompt"])

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
        result = validate_tool_call_detailed(response, item["schema"])
        if result.is_valid:
            print("STATUS: ✓ Strict-schema valid")
        else:
            print(f"STATUS: ✗ {result.error_type} - {result.message}")
        print()

if __name__ == "__main__":
    main()
