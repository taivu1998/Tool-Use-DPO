import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.dataset import read_jsonl
from src.prompts import build_chat_prompt
from src.validation import VALIDATION_MODE_STRICT, canonicalize_json, validate_tool_call_detailed


def _safe_expected_json(chosen: Any) -> Optional[str]:
    try:
        if isinstance(chosen, str):
            return canonicalize_json(json.loads(chosen))
        return canonicalize_json(chosen)
    except Exception:
        return None


def run_evaluation(
    model: Any,
    tokenizer: Any,
    data_path: str,
    max_new_tokens: int = 128,
    mode: str = VALIDATION_MODE_STRICT,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = read_jsonl(data_path)
    predictions: List[Dict[str, Any]] = []

    passed = 0
    exact_matches = 0
    valid_json = 0
    error_counts: Dict[str, int] = {}

    for index, row in enumerate(tqdm(rows), start=1):
        prompt = build_chat_prompt(row["prompt"])
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]

        result = validate_tool_call_detailed(response, row["schema"], mode=mode)
        expected_json = _safe_expected_json(row.get("chosen"))

        if result.parsed_json is not None:
            valid_json += 1
        if result.is_valid:
            passed += 1
        else:
            error_counts[result.error_type] = error_counts.get(result.error_type, 0) + 1

        if result.normalized_json is not None and expected_json is not None and result.normalized_json == expected_json:
            exact_matches += 1

        predictions.append(
            {
                "index": index,
                "prompt": row["prompt"],
                "response": response,
                "is_valid": result.is_valid,
                "error_type": result.error_type,
                "message": result.message,
                "normalized_json": result.normalized_json,
                "expected_json": expected_json,
                "exact_match": result.normalized_json is not None and expected_json is not None and result.normalized_json == expected_json,
            }
        )

    total = len(rows)
    summary = {
        "data_path": data_path,
        "total_samples": total,
        "passed": passed,
        "valid_json": valid_json,
        "exact_matches": exact_matches,
        "strict_schema_pass_rate": passed / total if total else 0.0,
        "valid_json_rate": valid_json / total if total else 0.0,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "failure_breakdown": error_counts,
    }
    return summary, predictions


def save_evaluation_artifacts(
    output_dir: str,
    summary: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    create_run_subdir: bool = True,
) -> Dict[str, str]:
    metadata = metadata or {}
    run_dir = output_dir
    if create_run_subdir:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        suffix = 0
        candidate = os.path.join(output_dir, timestamp)
        while os.path.exists(candidate):
            suffix += 1
            candidate = os.path.join(output_dir, f"{timestamp}_{suffix:02d}")
        run_dir = candidate
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "output_dir": run_dir,
        **metadata,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    predictions_path = os.path.join(run_dir, "predictions.jsonl")
    failures_path = os.path.join(run_dir, "failure_cases.jsonl")
    manifest_path = os.path.join(run_dir, "run_manifest.json")

    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    with open(predictions_path, "w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with open(failures_path, "w") as handle:
        for row in predictions:
            if not row["is_valid"]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return {
        "output_dir": run_dir,
        "summary": summary_path,
        "predictions": predictions_path,
        "failures": failures_path,
        "manifest": manifest_path,
    }
