import json
import tempfile
import unittest

from src.dataset_audit import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def test_audit_reports_duplicates_and_open_object_nodes(self) -> None:
        rows = [
            {
                "prompt": "Create a task",
                "chosen": '{"tool":"create_task","parameters":{"title":"Review PR"}}',
                "rejected": '{"tool":"create_task","parameters":{"title":"Review PR","priority":"high"}}',
                "schema": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}},
                            "required": ["title"],
                        },
                    },
                    "required": ["tool", "parameters"],
                },
            },
            {
                "prompt": "Create a task",
                "chosen": '{"tool":"create_task","parameters":{"title":"Ship it"}}',
                "rejected": '{"tool":"create_task"}',
                "schema": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}},
                            "required": ["title"],
                        },
                    },
                    "required": ["tool", "parameters"],
                },
            },
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            data_path = handle.name

        report = audit_dataset(data_path)

        self.assertEqual(report["total_samples"], 2)
        self.assertEqual(report["valid_pairs"], 2)
        self.assertEqual(report["invalid_pairs"], 0)
        self.assertEqual(report["duplicate_prompt_count"], 1)
        self.assertGreater(report["open_object_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
