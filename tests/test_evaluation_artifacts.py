import tempfile
import unittest

from src.evaluation import save_evaluation_artifacts


class EvaluationArtifactTests(unittest.TestCase):
    def test_artifacts_use_distinct_run_directories(self) -> None:
        summary = {"total_samples": 1}
        predictions = [{"is_valid": True, "response": "{}"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            first = save_evaluation_artifacts(tmpdir, summary, predictions)
            second = save_evaluation_artifacts(tmpdir, summary, predictions)
            self.assertNotEqual(first["output_dir"], second["output_dir"])


if __name__ == "__main__":
    unittest.main()
