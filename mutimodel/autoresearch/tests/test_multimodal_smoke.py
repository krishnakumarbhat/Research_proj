from __future__ import annotations

import unittest

from multimodal_bench.run_all import execute


class MultimodalSmokeTest(unittest.TestCase):
    def test_quick_synthetic_run(self) -> None:
        project_results, artifacts = execute(quick=True, prefer_real_audio=False)
        self.assertEqual(len(project_results), 4)
        self.assertTrue(artifacts["report_md"].exists())
        self.assertTrue(artifacts["records_csv"].exists())
        self.assertTrue(artifacts["summary_csv"].exists())


if __name__ == "__main__":
    unittest.main()
