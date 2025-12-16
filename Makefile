.PHONY: install clean data sft dpo eval eval-baseline test validate-data

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

sft:
	python scripts/train_sft.py --config configs/sft_config.yaml

dpo:
	python scripts/train_dpo.py --config configs/dpo_config.yaml

eval:
	python scripts/evaluate.py --model_path checkpoints/dpo_final --data_path data/synthetic_triplets.jsonl

eval-baseline:
	python scripts/evaluate_baseline.py --data_path data/synthetic_triplets.jsonl

# Run full comparison (baseline vs DPO)
compare:
	@echo "=== Evaluating Baseline Model ==="
	python scripts/evaluate_baseline.py --data_path data/synthetic_triplets.jsonl
	@echo ""
	@echo "=== Evaluating DPO Model ==="
	python scripts/evaluate.py --model_path checkpoints/dpo_final --data_path data/synthetic_triplets.jsonl

# Validate generated data
validate-data:
	python -c "import json; from src.validation import validate_tool_call; \
	data = [json.loads(l) for l in open('data/synthetic_triplets.jsonl')]; \
	valid = sum(1 for d in data if validate_tool_call(d['chosen'], d['schema'])[0] and not validate_tool_call(d['rejected'], d['schema'])[0]); \
	print(f'Valid samples: {valid}/{len(data)}')"

# Test with sample data (no GPU required)
test:
	python -c "from src.utils import setup_logging, get_device; from src.validation import validate_tool_call; print('Imports OK'); print(f'Device: {get_device()}')"
	python -c "import json; from src.validation import validate_tool_call; \
	data = [json.loads(l) for l in open('data/sample_triplets.jsonl')]; \
	valid = sum(1 for d in data if validate_tool_call(d['chosen'], d['schema'])[0] and not validate_tool_call(d['rejected'], d['schema'])[0]); \
	assert valid == len(data), f'Expected all valid, got {valid}/{len(data)}'; \
	print(f'Sample data validation: {valid}/{len(data)} PASSED')"