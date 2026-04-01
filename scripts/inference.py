import argparse
import json

from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.model import load_model_and_tokenizer, prepare_model_for_inference
from src.prompts import build_chat_prompt
from src.validation import VALIDATION_MODE_DEBUG_EXTRACT, VALIDATION_MODE_STRICT, validate_tool_call_detailed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--schema_path", type=str, default=None)
    parser.add_argument("--debug_extract", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    prepare_model_for_inference(model)

    prompt_text = build_chat_prompt(args.prompt)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode only the generated tokens (excluding the input prompt)
    response = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    print("-" * 20)
    print("Response:")
    print(response.strip())
    print("-" * 20)

    if args.schema_path:
        with open(args.schema_path, "r") as handle:
            schema = json.load(handle)
        mode = VALIDATION_MODE_DEBUG_EXTRACT if args.debug_extract else VALIDATION_MODE_STRICT
        result = validate_tool_call_detailed(response, schema, mode=mode)
        print("Validation:")
        print(f"  valid: {result.is_valid}")
        print(f"  error_type: {result.error_type or 'none'}")
        print(f"  message: {result.message or 'ok'}")

if __name__ == "__main__":
    main()
