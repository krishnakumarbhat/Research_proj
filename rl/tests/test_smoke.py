from __future__ import annotations

import unittest

from rlbench.run_all import PROJECT_RUNNERS, execute


class BenchmarkSmokeTest(unittest.TestCase):
    def test_registry_contains_all_projects(self) -> None:
        self.assertEqual(len(PROJECT_RUNNERS), 15)
        self.assertIn("curriculum_jssp", PROJECT_RUNNERS)
        self.assertIn("mapf_optimization", PROJECT_RUNNERS)

    def test_selected_quick_projects_run(self) -> None:
        results, failures, artifacts = execute(
            quick=True,
            selected_projects=[
                "dynamic_pricing",
                "curriculum_jssp",
                "sim_to_real_transfer",
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