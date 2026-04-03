from __future__ import annotations

import unittest

from dl.run_all import PROJECT_RUNNERS, execute


class BenchmarkSmokeTest(unittest.TestCase):
    def test_registry_contains_all_projects(self) -> None:
        self.assertEqual(len(PROJECT_RUNNERS), 15)
        self.assertIn("sensor_fusion_architectures", PROJECT_RUNNERS)
        self.assertIn("adversarial_edge_robustness", PROJECT_RUNNERS)

    def test_selected_quick_projects_run(self) -> None:
        results, failures, artifacts = execute(
            quick=True,
            selected_projects=[
                "sensor_fusion_architectures",
                "quantization_time_series",
                "physics_informed_pinn",
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
