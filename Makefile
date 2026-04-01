.PHONY: install clean data prepare-splits sft dpo eval eval-baseline test validate-data audit-data

install:
	pip install -e .

install-unsloth:
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf logs/* checkpoints/*

# Execution Pipeline
data:
	python scripts/generate_data.py --output_file data/synthetic_triplets.jsonl --num_samples 500
	python scripts/prepare_dataset_splits.py --source_path data/synthetic_triplets.jsonl

prepare-splits:
	python scripts/prepare_dataset_splits.py --source_path data/synthetic_triplets.jsonl

sft:
	python scripts/train_sft.py --config configs/sft_config.yaml

dpo:
	python scripts/train_dpo.py --config configs/dpo_config.yaml

eval:
	python scripts/evaluate.py --model_path checkpoints/dpo_final --data_path data/test_triplets.jsonl

eval-baseline:
	python scripts/evaluate_baseline.py --data_path data/test_triplets.jsonl

# Run full comparison (baseline vs DPO)
compare:
	@echo "=== Evaluating Baseline Model ==="
	python scripts/evaluate_baseline.py --data_path data/test_triplets.jsonl
	@echo ""
	@echo "=== Evaluating DPO Model ==="
	python scripts/evaluate.py --model_path checkpoints/dpo_final --data_path data/test_triplets.jsonl

# Validate generated data
validate-data:
	python scripts/validate_dataset.py --data_path data/train_triplets.jsonl --fail_on_errors --fail_on_open_objects --fail_on_duplicate_prompts --fail_on_duplicate_pairs
	python scripts/validate_dataset.py --data_path data/val_triplets.jsonl --fail_on_errors --fail_on_open_objects --fail_on_duplicate_prompts --fail_on_duplicate_pairs
	python scripts/validate_dataset.py --data_path data/test_triplets.jsonl --fail_on_errors --fail_on_open_objects --fail_on_duplicate_prompts --fail_on_duplicate_pairs

audit-data:
	python scripts/validate_dataset.py --data_path data/synthetic_triplets.jsonl --report_path logs/dataset_audit_raw.json
	python scripts/validate_dataset.py --data_path data/train_triplets.jsonl --report_path logs/dataset_audit_train.json
	python scripts/validate_dataset.py --data_path data/val_triplets.jsonl --report_path logs/dataset_audit_val.json
	python scripts/validate_dataset.py --data_path data/test_triplets.jsonl --report_path logs/dataset_audit_test.json

# Test with sample data (no GPU required)
test:
	python -c "from src.utils import setup_logging, get_device; from src.validation import validate_tool_call; print('Imports OK'); print(f'Device: {get_device()}')"
	python scripts/validate_dataset.py --data_path data/sample_triplets.jsonl --fail_on_errors
	python -m unittest discover -s tests -p 'test_*.py'
