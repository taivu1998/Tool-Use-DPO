import json
import jsonschema
import logging
import re
from typing import Tuple, Dict, Any, Optional

def extract_json(text: str) -> Optional[str]:
    """
    Try to extract JSON from model output that may contain extra text.
    Handles common cases like markdown code blocks or prefixed/suffixed text.
    Returns the extracted JSON string or None if extraction fails.
    """
    text = text.strip()

    # Case 1: Already valid JSON
    if text.startswith('{') and text.endswith('}'):
        return text

    # Case 2: Markdown code block ```json ... ```
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if md_match:
        return md_match.group(1)

    # Case 3: JSON embedded in text - find first { and matching }
    start = text.find('{')
    if start != -1:
        # Find the matching closing brace
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

    return None

def validate_tool_call(json_str: str, schema: Dict[str, Any], strict: bool = True) -> Tuple[bool, str]:
    """
    Validates a JSON string against a tool schema.

    Args:
        json_str: The string to validate (may contain extra text if strict=False)
        schema: JSON schema to validate against
        strict: If False, try to extract JSON from text that contains extra content

    Returns: (is_valid, error_message)
    """
    text_to_validate = json_str

    # If not strict, try to extract JSON from the response
    if not strict:
        extracted = extract_json(json_str)
        if extracted:
            text_to_validate = extracted

    try:
        data = json.loads(text_to_validate)
        jsonschema.validate(instance=data, schema=schema)
        return True, ""
    except json.JSONDecodeError:
        return False, f"Invalid JSON Syntax (raw: {json_str[:100]}...)"
    except jsonschema.ValidationError as e:
        return False, f"Schema Violation: {e.message}"
    except Exception as e:
        return False, f"Unknown Error: {str(e)}"