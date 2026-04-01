import tempfile
import unittest

from src.config_parser import parse_args_with_config


class ConfigParserTests(unittest.TestCase):
    def test_cli_overrides_are_coerced_safely(self) -> None:
        config_text = """
flag: true
epochs: 3
learning_rate: 1.0e-4
tags:
  - alpha
  - beta
optional_value: null
name: base
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        parsed = parse_args_with_config(
            [
                "--config",
                config_path,
                "--flag",
                "false",
                "--epochs",
                "8",
                "--learning_rate",
                "0.01",
                "--tags",
                '["gamma", "delta"]',
                "--optional_value",
                "42",
                "--name",
                "refined",
            ]
        )

        self.assertEqual(parsed["flag"], False)
        self.assertEqual(parsed["epochs"], 8)
        self.assertEqual(parsed["learning_rate"], 0.01)
        self.assertEqual(parsed["tags"], ["gamma", "delta"])
        self.assertEqual(parsed["optional_value"], 42)
        self.assertEqual(parsed["name"], "refined")


if __name__ == "__main__":
    unittest.main()
