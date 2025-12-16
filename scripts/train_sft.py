import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from src.model import load_model_and_tokenizer, prepare_model_for_peft
from src.dataset import load_sft_dataset
from src.config_parser import parse_args_with_config
from src.utils import setup_logging, seed_everything

def main():
    args = parse_args_with_config()
    setup_logging(args.get("log_file"))
    seed_everything(args.get("seed", 42))

    model, tokenizer = load_model_and_tokenizer(
        model_name=args["model_name"],
        max_seq_length=args["max_seq_length"]
    )

    # Fix tokenizer for Qwen/Unsloth/TRL compatibility
    eos_token = "<|im_end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    tokenizer.eos_token = eos_token
    tokenizer.eos_token_id = eos_token_id
    tokenizer.pad_token = eos_token
    tokenizer.pad_token_id = eos_token_id
    tokenizer.padding_side = "right"
    model.config.eos_token_id = eos_token_id
    model.config.pad_token_id = eos_token_id

    model = prepare_model_for_peft(model)

    dataset = load_sft_dataset(args["data_path"], tokenizer)

    # Pre-tokenize the dataset
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args["max_seq_length"],
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Create data collator for padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args["output_dir"],
        per_device_train_batch_size=args["batch_size"],
        gradient_accumulation_steps=args["grad_accum_steps"],
        warmup_steps=args["warmup_steps"],
        num_train_epochs=args["epochs"],
        learning_rate=args["learning_rate"],
        weight_decay=args.get("weight_decay", 0.01),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        seed=args["seed"],
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args["output_dir"])
    tokenizer.save_pretrained(args["output_dir"])

if __name__ == "__main__":
    main()
