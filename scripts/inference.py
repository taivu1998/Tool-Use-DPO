from unsloth import FastLanguageModel
import argparse
from src.utils import get_device

SYSTEM_PROMPT = """You are a tool-calling assistant. When given a user request and tool specification, respond with ONLY a valid JSON object representing the tool call. Do not include any explanation, markdown formatting, or code blocks. Output raw JSON only."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = get_device()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

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

if __name__ == "__main__":
    main()