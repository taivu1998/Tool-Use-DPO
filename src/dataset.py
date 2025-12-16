import json
from datasets import Dataset
import logging

SYSTEM_PROMPT = """You are a tool-calling assistant. When given a user request and tool specification, respond with ONLY a valid JSON object representing the tool call. Do not include any explanation, markdown formatting, or code blocks. Output raw JSON only."""

def format_dpo_pair(example):
    """
    Formats the triplet for ChatML.
    Qwen-2.5 expects specific chat templates.
    """
    # Include system prompt to constrain output format
    prompt_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"

    return {
        "prompt": prompt_text,
        "chosen": example["chosen"] + "<|im_end|>",
        "rejected": example["rejected"] + "<|im_end|>"
    }

def load_dpo_dataset(data_path: str, tokenizer=None):
    """Loads JSONL data and prepares it for DPO training."""
    logging.info(f"Loading dataset from {data_path}")
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    dataset = Dataset.from_list(data)
    
    # Map to ChatML format
    dataset = dataset.map(format_dpo_pair)
    
    return dataset

def load_sft_dataset(data_path: str, tokenizer):
    """
    Loads JSONL data for SFT Cold Start.
    Uses only 'prompt' + 'chosen'.
    """
    logging.info(f"Loading SFT dataset from {data_path}")

    def format_sft(example):
        # Include system prompt to teach the model output format
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['chosen']}<|im_end|>"
        return {"text": text}

    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_sft)
    return dataset