import json
import logging
from typing import Any, Dict, List, Optional

from datasets import Dataset

from src.prompts import append_assistant_end, format_dpo_prompt, format_sft_text


def read_jsonl(data_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into memory."""
    logging.info("Loading dataset from %s", data_path)
    data: List[Dict[str, Any]] = []
    with open(data_path, "r") as handle:
        for line in handle:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_dpo_pair(example: Dict[str, Any]) -> Dict[str, str]:
    """Format a preference triplet using the shared ChatML prompt."""
    return {
        "prompt": format_dpo_prompt(example["prompt"]),
        "chosen": append_assistant_end(example["chosen"]),
        "rejected": append_assistant_end(example["rejected"]),
    }


def load_dpo_dataset(data_path: str, tokenizer: Optional[Any] = None) -> Dataset:
    """Load JSONL data and prepare it for DPO training."""
    del tokenizer  # The signature is kept for backward compatibility.
    dataset = Dataset.from_list(read_jsonl(data_path))
    return dataset.map(format_dpo_pair)


def load_sft_dataset(data_path: str, tokenizer: Optional[Any] = None) -> Dataset:
    """
    Load JSONL data for SFT cold start using only ``prompt`` + ``chosen``.
    """
    del tokenizer

    def format_sft(example: Dict[str, Any]) -> Dict[str, str]:
        return {"text": format_sft_text(example["prompt"], example["chosen"])}

    dataset = Dataset.from_list(read_jsonl(data_path))
    return dataset.map(format_sft)
