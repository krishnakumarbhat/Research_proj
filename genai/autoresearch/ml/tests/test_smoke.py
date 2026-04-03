from __future__ import annotations

import unittest

from ml.run_all import PROJECT_RUNNERS, execute


class BenchmarkSmokeTests(unittest.TestCase):
    def test_project_registry_has_all_topics(self) -> None:
        self.assertEqual(len(PROJECT_RUNNERS), 15)

    def test_quick_subset_executes(self) -> None:
        results, failures, _ = execute(
            quick=True,
            selected_projects=[
                "constrained_generation_logit_masking",
                "edge_optimized_rag",
                "synthetic_data_generation_niche_classifiers",
            ],
            continue_on_error=False,
            write_outputs=False,
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(failures, [])
        for project in results:
            self.assertTrue(project.results)


if __name__ == "__main__":
    unittest.main()