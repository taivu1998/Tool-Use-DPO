import json
from collections import Counter
from typing import Any, Dict, List, Optional

from src.dataset import read_jsonl
from src.validation import (
    ERROR_TYPE_NONE,
    VALIDATION_MODE_STRICT,
    count_object_nodes,
    validate_tool_call_detailed,
)


def audit_dataset(
    data_path: str,
    mode: str = VALIDATION_MODE_STRICT,
) -> Dict[str, Any]:
    rows = read_jsonl(data_path)
    error_counts: Counter[str] = Counter()
    chosen_failures: List[Dict[str, Any]] = []
    rejected_failures: List[Dict[str, Any]] = []
    duplicate_prompt_count = len(rows) - len({row.get("prompt") for row in rows})
    duplicate_pair_count = len(rows) - len(
        {
            (
                row.get("prompt"),
                row.get("chosen"),
                row.get("rejected"),
            )
            for row in rows
        }
    )
    open_object_nodes = 0
    total_object_nodes = 0
    top_level_keys: Counter[str] = Counter()

    valid_pairs = 0
    for idx, row in enumerate(rows, start=1):
        schema = row.get("schema")
        if isinstance(schema, dict):
            top_level_keys.update(schema.get("properties", {}).keys())
            open_nodes, object_nodes = count_object_nodes(schema)
            open_object_nodes += open_nodes
            total_object_nodes += object_nodes

        chosen = validate_tool_call_detailed(row.get("chosen"), schema, mode=mode)
        rejected = validate_tool_call_detailed(row.get("rejected"), schema, mode=mode)

        if chosen.is_valid and not rejected.is_valid:
            valid_pairs += 1
        else:
            if not chosen.is_valid:
                error_counts[f"chosen:{chosen.error_type}"] += 1
                chosen_failures.append(
                    {
                        "index": idx,
                        "prompt": row.get("prompt"),
                        "error_type": chosen.error_type,
                        "message": chosen.message,
                    }
                )
            if rejected.is_valid:
                error_counts["rejected:unexpected_valid"] += 1
                rejected_failures.append(
                    {
                        "index": idx,
                        "prompt": row.get("prompt"),
                        "error_type": ERROR_TYPE_NONE,
                        "message": "Rejected sample unexpectedly passed validation.",
                    }
                )
            elif not rejected.is_valid:
                error_counts[f"rejected:{rejected.error_type}"] += 1

    return {
        "data_path": data_path,
        "total_samples": len(rows),
        "valid_pairs": valid_pairs,
        "invalid_pairs": len(rows) - valid_pairs,
        "duplicate_prompt_count": duplicate_prompt_count,
        "duplicate_pair_count": duplicate_pair_count,
        "open_object_nodes": open_object_nodes,
        "total_object_nodes": total_object_nodes,
        "top_level_property_keys": dict(top_level_keys.most_common()),
        "error_counts": dict(error_counts),
        "chosen_failures": chosen_failures,
        "rejected_failures": rejected_failures,
    }


def write_audit_report(report: Dict[str, Any], output_path: Optional[str]) -> None:
    if not output_path:
        return
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
