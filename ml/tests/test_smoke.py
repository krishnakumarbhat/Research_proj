from __future__ import annotations

import unittest

from ml.run_all import PROJECT_RUNNERS, execute


class BenchmarkSmokeTest(unittest.TestCase):
    def test_registry_contains_all_projects(self) -> None:
        self.assertEqual(len(PROJECT_RUNNERS), 15)
        self.assertIn("ebm_credit", PROJECT_RUNNERS)
        self.assertIn("agentic_data_validation", PROJECT_RUNNERS)

    def test_selected_quick_projects_run(self) -> None:
        results, failures, artifacts = execute(
            quick=True,
            selected_projects=[
                "tree_noise_robustness",
                "label_encoding_stability",
                "agentic_data_validation",
            ],
            continue_on_error=False,
            write_outputs=False,
        )
        self.assertFalse(failures)
        self.assertEqual(len(results), 3)
        self.assertFalse(artifacts)
        for result in results:
            self.assertTrue(result.records)


if __name__ == "__main__":
    unittest.main()