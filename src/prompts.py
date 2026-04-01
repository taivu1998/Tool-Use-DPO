SYSTEM_PROMPT = (
    "You are a tool-calling assistant. When given a user request and tool "
    "specification, respond with ONLY a valid JSON object representing the tool "
    "call. Do not include any explanation, markdown formatting, or code blocks. "
    "Output raw JSON only."
)

CHAT_TEMPLATE_PREFIX = (
    "<|im_start|>system\n"
    f"{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n"
)
CHAT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
CHAT_TEMPLATE_END = "<|im_end|>"


def build_chat_prompt(user_prompt: str) -> str:
    """Build the shared ChatML prompt used across train/eval/inference."""
    return f"{CHAT_TEMPLATE_PREFIX}{user_prompt}{CHAT_TEMPLATE_SUFFIX}"


def format_sft_text(user_prompt: str, chosen_response: str) -> str:
    """Format a full SFT example with the assistant response included."""
    return f"{build_chat_prompt(user_prompt)}{chosen_response}{CHAT_TEMPLATE_END}"


def format_dpo_prompt(user_prompt: str) -> str:
    """Format the prompt prefix for DPO preference training."""
    return build_chat_prompt(user_prompt)


def append_assistant_end(text: str) -> str:
    """Append the shared assistant terminator token."""
    return f"{text}{CHAT_TEMPLATE_END}"
