from __future__ import annotations

from pathlib import Path

from cv_bench.run_all import PROJECT_RUNNERS, execute
from cv_bench.run_all_project_files import _discover_project_files


def test_registry_contains_all_projects() -> None:
    assert len(PROJECT_RUNNERS) == 15


def test_project_files_match_registry() -> None:
    assert len(_discover_project_files()) == len(PROJECT_RUNNERS)


def test_quick_execute_selected_projects_writes_artifacts(tmp_path: Path) -> None:
    selected = [
        "event_camera_processing",
        "visual_anomaly_patchcore",
        "gaze_tracking",
    ]
    results, failures, artifacts = execute(
        quick=True,
        selected_projects=selected,
        continue_on_error=False,
        write_outputs=True,
        artifact_root=tmp_path,
    )

    assert not failures
    assert [result.project for result in results] == selected
    assert artifacts["records_csv"].exists()
    assert artifacts["summary_csv"].exists()
    report_text = artifacts["report_md"].read_text(encoding="utf-8")
    assert "CV Research Benchmark Report" in report_text
    assert "Event Camera (Neuromorphic) Processing" in report_text