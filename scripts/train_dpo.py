import glob
import torch
from trl import DPOTrainer, DPOConfig
from _bootstrap import PROJECT_ROOT  # noqa: F401
from src.dataset import load_dpo_dataset
from src.config_parser import parse_args_with_config
from src.model import load_model_and_tokenizer, prepare_model_for_training
from src.utils import setup_logging, seed_everything

def main():
    args = parse_args_with_config()
    setup_logging(args.get("log_file"))
    seed_everything(args.get("seed", 42))

    # Load the SFT Cold Start model
    model, tokenizer = load_model_and_tokenizer(
        model_name=args["model_name"],
        max_seq_length=args["max_seq_length"],
        load_in_4bit=True,
    )

    # Ensure training mode is set
    prepare_model_for_training(model)

    dataset = load_dpo_dataset(args["data_path"], tokenizer)

    # Check for existing checkpoint to resume from
    resume_from_checkpoint = None
    checkpoint_dirs = glob.glob(f"{args['output_dir']}/checkpoint-*")
    if checkpoint_dirs:
        resume_from_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
        print(f"Found checkpoint to resume from: {resume_from_checkpoint}")
    else:
        print("No checkpoint found, training from scratch")

    # Create DPO config with fixes
    dpo_args = DPOConfig(
        per_device_train_batch_size=args["batch_size"],
        gradient_accumulation_steps=args["grad_accum_steps"],
        warmup_ratio=0.1,
        num_train_epochs=args["epochs"],
        learning_rate=args["learning_rate"],
        weight_decay=args.get("weight_decay", 0.01),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        beta=args["beta"],
        loss_type=args["loss_type"],
        max_length=args["max_seq_length"],
        max_prompt_length=args.get("max_prompt_length", 768),
        output_dir=args["output_dir"],
        optim="adamw_8bit",
        seed=args["seed"],
        report_to="none",
        dataset_num_proc=1,
        # Checkpoint settings
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
    )

    # Override eos_token to skip TRL's problematic check
    dpo_args.eos_token = None

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=dpo_args,
    )

    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
        dpo_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        dpo_trainer.train()
    dpo_trainer.save_model(args["output_dir"])
    tokenizer.save_pretrained(args["output_dir"])

if __name__ == "__main__":
    main()
