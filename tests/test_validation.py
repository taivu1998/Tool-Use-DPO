import unittest

from src.validation import (
    ERROR_TYPE_FORMAT_VIOLATION,
    ERROR_TYPE_HALLUCINATED_PARAM,
    VALIDATION_MODE_DEBUG_EXTRACT,
    validate_tool_call,
    validate_tool_call_detailed,
)


class ValidationTests(unittest.TestCase):
    def test_format_date_is_enforced(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "date": {"type": "string", "format": "date"},
            },
            "required": ["tool", "date"],
        }

        valid_result = validate_tool_call_detailed('{"tool":"x","date":"2024-03-15"}', schema)
        invalid_result = validate_tool_call_detailed('{"tool":"x","date":"not-a-date"}', schema)

        self.assertTrue(valid_result.is_valid)
        self.assertFalse(invalid_result.is_valid)
        self.assertEqual(invalid_result.error_type, ERROR_TYPE_FORMAT_VIOLATION)

    def test_missing_additional_properties_defaults_to_strict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
            "required": ["tool", "parameters"],
        }

        result = validate_tool_call_detailed(
            '{"tool":"get_weather","parameters":{"city":"SF","units":"celsius"}}',
            schema,
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_type, ERROR_TYPE_HALLUCINATED_PARAM)

    def test_debug_extract_mode_recovers_wrapped_json(self) -> None:
        schema = {
            "type": "object",
            "properties": {"tool": {"type": "string"}},
            "required": ["tool"],
        }

        result = validate_tool_call_detailed(
            "```json\n{\"tool\":\"search\"}\n```",
            schema,
            mode=VALIDATION_MODE_DEBUG_EXTRACT,
        )
        self.assertTrue(result.is_valid)

        strict_valid, strict_message = validate_tool_call("```json\n{\"tool\":\"search\"}\n```", schema, strict=True)
        self.assertFalse(strict_valid)
        self.assertIn("Invalid JSON syntax", strict_message)


if __name__ == "__main__":
    unittest.main()
