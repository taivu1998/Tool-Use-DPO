import json
import tempfile
import unittest

from src.dataset_split import prepare_dataset_splits


class DatasetSplitTests(unittest.TestCase):
    def test_prepare_dataset_splits_filters_invalid_rows_and_dedupes(self) -> None:
        rows = [
            {
                "prompt": "Create task A",
                "chosen": '{"tool":"create_task","parameters":{"title":"A"}}',
                "rejected": '{"tool":"create_task","parameters":{"title":"A","priority":"high"}}',
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
                "prompt": "Create task A",
                "chosen": '{"tool":"create_task","parameters":{"title":"A"}}',
                "rejected": '{"tool":"create_task","parameters":{"title":"A","priority":"high"}}',
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
                "prompt": "Bad chosen row",
                "chosen": '{"tool":"x","date":"bad-date"}',
                "rejected": '{"tool":"x"}',
                "schema": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "date": {"type": "string", "format": "date"},
                    },
                    "required": ["tool", "date"],
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = f"{tmpdir}/source.jsonl"
            train_path = f"{tmpdir}/train.jsonl"
            val_path = f"{tmpdir}/val.jsonl"
            test_path = f"{tmpdir}/test.jsonl"

            with open(source_path, "w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            report = prepare_dataset_splits(
                source_path=source_path,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                train_ratio=0.8,
                val_ratio=0.0,
                seed=1,
                dedupe_by="prompt",
            )

            self.assertEqual(report["source_rows"], 3)
            self.assertEqual(report["valid_rows"], 2)
            self.assertEqual(report["deduped_rows"], 1)
            self.assertEqual(report["train_rows"] + report["val_rows"] + report["test_rows"], 1)


if __name__ == "__main__":
    unittest.main()
