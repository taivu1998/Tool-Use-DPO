import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

VALIDATION_MODE_STRICT = "strict"
VALIDATION_MODE_COMPAT = "compat"
VALIDATION_MODE_DEBUG_EXTRACT = "debug_extract"

ERROR_TYPE_NONE = ""
ERROR_TYPE_JSON_SYNTAX = "json_error"
ERROR_TYPE_BAD_SCHEMA = "bad_schema"
ERROR_TYPE_HALLUCINATED_PARAM = "hallucinated_param"
ERROR_TYPE_TYPE_MISMATCH = "type_mismatch"
ERROR_TYPE_ENUM_VIOLATION = "enum_violation"
ERROR_TYPE_MISSING_REQUIRED = "missing_required"
ERROR_TYPE_FORMAT_VIOLATION = "format_violation"
ERROR_TYPE_CONST_VIOLATION = "const_violation"
ERROR_TYPE_OTHER_SCHEMA_ERROR = "other_schema_error"
ERROR_TYPE_UNKNOWN = "unknown_error"

STRICT_OBJECT_OPT_OUT_KEY = "x-allow-additional-properties"

FORMAT_CHECKER = jsonschema.FormatChecker()


@dataclass
class ValidationResult:
    is_valid: bool
    error_type: str = ERROR_TYPE_NONE
    message: str = ""
    stage: str = "validation"
    normalized_json: Optional[str] = None
    parsed_json: Optional[Any] = None
    normalized_schema: Optional[Dict[str, Any]] = None


def extract_json(text: str) -> Optional[str]:
    """
    Try to extract JSON from model output that may contain extra text.
    Handles common cases like markdown code blocks or prefixed/suffixed text.
    Returns the extracted JSON string or None if extraction fails.
    """
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        return md_match.group(1)

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


def _coerce_json_string(text: Any) -> str:
    if isinstance(text, str):
        return text.strip()
    return json.dumps(text, sort_keys=True)


def canonicalize_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _object_schema_is_open(schema: Dict[str, Any]) -> bool:
    if schema.get(STRICT_OBJECT_OPT_OUT_KEY):
        return True
    if schema.get("additionalProperties") is not None:
        return bool(schema["additionalProperties"])
    if "patternProperties" in schema:
        return True
    return False


def normalize_schema(
    schema: Dict[str, Any],
    close_objects: bool = True,
) -> Dict[str, Any]:
    """
    Recursively normalize a JSON schema into a stricter, canonical form.

    By default, object schemas without an explicit openness declaration are
    treated as closed-world schemas via ``additionalProperties: false``.
    """
    normalized = deepcopy(schema)

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            visited = {key: visit(value) for key, value in node.items()}
            node_type = visited.get("type")
            if node_type == "object" and close_objects and not _object_schema_is_open(visited):
                visited["additionalProperties"] = False
            return visited
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return visit(normalized)


def count_object_nodes(schema: Dict[str, Any]) -> Tuple[int, int]:
    """Return (open_object_nodes, total_object_nodes) for audit/reporting."""
    open_nodes = 0
    total_nodes = 0

    def visit(node: Any) -> None:
        nonlocal open_nodes, total_nodes
        if isinstance(node, dict):
            if node.get("type") == "object":
                total_nodes += 1
                if (
                    node.get("additionalProperties") is not False
                    and STRICT_OBJECT_OPT_OUT_KEY not in node
                ):
                    open_nodes += 1
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return open_nodes, total_nodes


def parse_json(text: str) -> Tuple[Optional[Any], Optional[json.JSONDecodeError]]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, exc


def _instance_path(error: jsonschema.ValidationError) -> str:
    if not error.absolute_path:
        return "$"
    path_parts: List[str] = []
    for part in error.absolute_path:
        if isinstance(part, int):
            path_parts.append(f"[{part}]")
        else:
            prefix = "." if path_parts else "."
            path_parts.append(f"{prefix}{part}")
    return f"${''.join(path_parts)}"


def categorize_validation_error(error: jsonschema.ValidationError) -> str:
    if error.validator == "required":
        return ERROR_TYPE_MISSING_REQUIRED
    if error.validator == "additionalProperties":
        return ERROR_TYPE_HALLUCINATED_PARAM
    if error.validator == "type":
        return ERROR_TYPE_TYPE_MISMATCH
    if error.validator == "enum":
        return ERROR_TYPE_ENUM_VIOLATION
    if error.validator == "format":
        return ERROR_TYPE_FORMAT_VIOLATION
    if error.validator == "const":
        return ERROR_TYPE_CONST_VIOLATION
    return ERROR_TYPE_OTHER_SCHEMA_ERROR


def _make_message(error: jsonschema.ValidationError) -> str:
    return f"Schema violation at {_instance_path(error)}: {error.message}"


def _build_validator(schema: Dict[str, Any]) -> Any:
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema, format_checker=FORMAT_CHECKER)


def validate_tool_call_detailed(
    json_str: Any,
    schema: Dict[str, Any],
    mode: str = VALIDATION_MODE_STRICT,
    close_objects: bool = True,
) -> ValidationResult:
    """Validate a tool-call payload and return structured metadata."""
    if not isinstance(schema, dict):
        return ValidationResult(
            is_valid=False,
            error_type=ERROR_TYPE_BAD_SCHEMA,
            message="Schema must be a JSON object.",
            stage="schema",
        )

    normalized_schema = normalize_schema(schema, close_objects=close_objects)
    raw_text = _coerce_json_string(json_str)
    text_to_validate = raw_text

    if mode == VALIDATION_MODE_DEBUG_EXTRACT:
        extracted = extract_json(raw_text)
        if extracted:
            text_to_validate = extracted

    parsed_json, json_error = parse_json(text_to_validate)
    if json_error is not None:
        return ValidationResult(
            is_valid=False,
            error_type=ERROR_TYPE_JSON_SYNTAX,
            message=f"Invalid JSON syntax: {json_error.msg}",
            stage="json",
            normalized_schema=normalized_schema,
        )

    try:
        validator = _build_validator(normalized_schema)
    except jsonschema.SchemaError as exc:
        return ValidationResult(
            is_valid=False,
            error_type=ERROR_TYPE_BAD_SCHEMA,
            message=f"Invalid schema: {exc.message}",
            stage="schema",
            parsed_json=parsed_json,
            normalized_json=canonicalize_json(parsed_json),
            normalized_schema=normalized_schema,
        )

    errors = sorted(validator.iter_errors(parsed_json), key=lambda err: list(err.absolute_path))
    if errors:
        first_error = errors[0]
        return ValidationResult(
            is_valid=False,
            error_type=categorize_validation_error(first_error),
            message=_make_message(first_error),
            stage="schema",
            parsed_json=parsed_json,
            normalized_json=canonicalize_json(parsed_json),
            normalized_schema=normalized_schema,
        )

    return ValidationResult(
        is_valid=True,
        normalized_json=canonicalize_json(parsed_json),
        parsed_json=parsed_json,
        normalized_schema=normalized_schema,
    )


def validate_tool_call(json_str: Any, schema: Dict[str, Any], strict: bool = True) -> Tuple[bool, str]:
    """
    Backward-compatible wrapper for callers that expect ``(bool, message)``.
    """
    mode = VALIDATION_MODE_STRICT if strict else VALIDATION_MODE_DEBUG_EXTRACT
    result = validate_tool_call_detailed(json_str, schema, mode=mode)
    return result.is_valid, result.message
