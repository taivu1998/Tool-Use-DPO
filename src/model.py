import torch
from unsloth import FastLanguageModel
import logging

def load_model_and_tokenizer(
    model_name: str, 
    max_seq_length: int = 2048, 
    load_in_4bit: bool = True
):
    """
    Wrapper for Unsloth's FastLanguageModel.
    Optimized for Qwen-2.5-Coder and A10G/3090 hardware.
    """
    logging.info(f"Loading Unsloth model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect (Float16 or Bfloat16)
        load_in_4bit=load_in_4bit,
    )
    
    return model, tokenizer

def prepare_model_for_peft(model):
    """Configures LoRA adapters for training."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,                # Increased from 16 for more capacity
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,       # 2x rank for optimal scaling
        lora_dropout=0,      # 0 is optimized for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,     # Rank-stabilized LoRA for better training
        loftq_config=None,
    )
    return model