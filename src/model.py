import logging
from typing import Any, Tuple

QWEN_CHAT_EOS_TOKEN = "<|im_end|>"


def _get_fast_language_model() -> Any:
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required for model loading. Install it with `make install-unsloth`."
        ) from exc
    return FastLanguageModel


def configure_special_tokens(model: Any, tokenizer: Any, eos_token: str = QWEN_CHAT_EOS_TOKEN) -> Tuple[Any, Any]:
    """Apply the shared Qwen/Unsloth token configuration."""
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_token_id is None or eos_token_id < 0:
        raise ValueError(f"Tokenizer does not recognize EOS token: {eos_token}")

    tokenizer.eos_token = eos_token
    tokenizer.eos_token_id = eos_token_id
    tokenizer.pad_token = eos_token
    tokenizer.pad_token_id = eos_token_id
    tokenizer.padding_side = "right"

    model.config.eos_token_id = eos_token_id
    model.config.pad_token_id = eos_token_id
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.eos_token_id = eos_token_id
        generation_config.pad_token_id = eos_token_id

    return model, tokenizer


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """
    Wrapper for Unsloth's FastLanguageModel with shared token setup.
    """
    FastLanguageModel = _get_fast_language_model()
    logging.info("Loading Unsloth model: %s", model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    return configure_special_tokens(model, tokenizer)


def prepare_model_for_training(model: Any) -> Any:
    FastLanguageModel = _get_fast_language_model()
    FastLanguageModel.for_training(model)
    return model


def prepare_model_for_inference(model: Any) -> Any:
    FastLanguageModel = _get_fast_language_model()
    FastLanguageModel.for_inference(model)
    return model


def prepare_model_for_peft(model: Any) -> Any:
    """Configure LoRA adapters for training."""
    FastLanguageModel = _get_fast_language_model()
    return FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )
