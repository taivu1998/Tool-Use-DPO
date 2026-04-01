import json
import os
import random
from typing import Any, Dict, List

from src.dataset import read_jsonl
from src.validation import VALIDATION_MODE_STRICT, validate_tool_call_detailed


def _dedupe_rows(rows: List[Dict[str, Any]], dedupe_by: str) -> List[Dict[str, Any]]:
    if dedupe_by == "none":
        return rows

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        if dedupe_by == "prompt":
            key = row.get("prompt")
        elif dedupe_by == "pair":
            key = (row.get("prompt"), row.get("chosen"), row.get("rejected"))
        else:
            raise ValueError(f"Unsupported dedupe mode: {dedupe_by}")

        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _normalize_valid_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        schema = row.get("schema")
        chosen = validate_tool_call_detailed(row.get("chosen"), schema, mode=VALIDATION_MODE_STRICT)
        rejected = validate_tool_call_detailed(row.get("rejected"), schema, mode=VALIDATION_MODE_STRICT)
        if chosen.is_valid and not rejected.is_valid:
            normalized_rows.append(
                {
                    **row,
                    "schema": chosen.normalized_schema,
                    "chosen": chosen.normalized_json or row.get("chosen"),
                    "rejected": rejected.normalized_json or row.get("rejected"),
                }
            )
    return normalized_rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_dataset_splits(
    source_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    dedupe_by: str = "prompt",
) -> Dict[str, Any]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected ratios where train > 0, val >= 0, and train + val < 1.")

    source_rows = read_jsonl(source_path)
    valid_rows = _normalize_valid_rows(source_rows)
    deduped_rows = _dedupe_rows(valid_rows, dedupe_by=dedupe_by)

    rng = random.Random(seed)
    rng.shuffle(deduped_rows)

    total = len(deduped_rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_rows = deduped_rows[:train_end]
    val_rows = deduped_rows[train_end:val_end]
    test_rows = deduped_rows[val_end:]

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(test_path, test_rows)

    return {
        "source_path": source_path,
        "source_rows": len(source_rows),
        "valid_rows": len(valid_rows),
        "deduped_rows": len(deduped_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "dedupe_by": dedupe_by,
        "seed": seed,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
    }
