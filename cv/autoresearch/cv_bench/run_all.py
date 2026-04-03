from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from cv_bench.common import ProjectResult, ROOT, ensure_directories
from cv_bench.projects.project_climate_patch_segmentation import run as run_climate_patch_segmentation
from cv_bench.projects.project_constrained_depth_estimation import run as run_constrained_depth_estimation
from cv_bench.projects.project_document_layout_analysis import run as run_document_layout_analysis
from cv_bench.projects.project_edge_semantic_segmentation import run as run_edge_semantic_segmentation
from cv_bench.projects.project_event_camera import run as run_event_camera
from cv_bench.projects.project_gaze_tracking import run as run_gaze_tracking
from cv_bench.projects.project_hybrid_edge_tracking import run as run_hybrid_edge_tracking
from cv_bench.projects.project_hyperspectral_agriculture import run as run_hyperspectral_agriculture
from cv_bench.projects.project_medical_ultrasound_segmentation import run as run_medical_ultrasound_segmentation
from cv_bench.projects.project_micro_expression_recognition import run as run_micro_expression_recognition
from cv_bench.projects.project_multicamera_calibration import run as run_multicamera_calibration
from cv_bench.projects.project_procedural_edge_case_generation import run as run_procedural_edge_case_generation
from cv_bench.projects.project_sensor_degraded_tracking import run as run_sensor_degraded_tracking
from cv_bench.projects.project_synthetic_graphics_validation import run as run_synthetic_graphics_validation
from cv_bench.projects.project_visual_anomaly_patchcore import run as run_visual_anomaly_patchcore
from cv_bench.reporting import write_csv_outputs, write_markdown_report


PROJECT_RUNNERS: dict[str, Callable[..., ProjectResult]] = {
    "event_camera_processing": run_event_camera,
    "sensor_degraded_object_tracking": run_sensor_degraded_tracking,
    "edge_compute_semantic_segmentation": run_edge_semantic_segmentation,
    "multicamera_calibration": run_multicamera_calibration,
    "procedural_synthetic_edge_cases": run_procedural_edge_case_generation,
    "visual_anomaly_patchcore": run_visual_anomaly_patchcore,
    "hyperspectral_agriculture": run_hyperspectral_agriculture,
    "document_layout_analysis": run_document_layout_analysis,
    "micro_expression_recognition": run_micro_expression_recognition,
    "medical_ultrasound_segmentation": run_medical_ultrasound_segmentation,
    "gaze_tracking": run_gaze_tracking,
    "synthetic_graphics_validation": run_synthetic_graphics_validation,
    "constrained_depth_estimation": run_constrained_depth_estimation,
    "hybrid_edge_tracking": run_hybrid_edge_tracking,
    "climate_patch_segmentation": run_climate_patch_segmentation,
}


def execute(
    *,
    quick: bool = True,
    selected_projects: list[str] | None = None,
    continue_on_error: bool = False,
    write_outputs: bool = True,
    artifact_root: Path | None = None,
) -> tuple[list[ProjectResult], list[tuple[str, str]], dict[str, Path]]:
    ensure_directories()
    selected = selected_projects or list(PROJECT_RUNNERS.keys())
    results: list[ProjectResult] = []
    failures: list[tuple[str, str]] = []

    for project in selected:
        if project not in PROJECT_RUNNERS:
            message = f"Unknown project id: {project}"
            if continue_on_error:
                failures.append((project, message))
                continue
            raise KeyError(message)
        try:
            results.append(PROJECT_RUNNERS[project](quick=quick))
        except Exception as exc:
            failures.append((project, str(exc)))
            if not continue_on_error:
                raise

    artifacts: dict[str, Path] = {}
    if write_outputs and results:
        output_root = artifact_root or ROOT
        records_path, summary_path = write_csv_outputs(results, output_root / "results")
        report_path = write_markdown_report(results, output_root / "research_report.md")
        artifacts = {
            "records_csv": records_path,
            "summary_csv": summary_path,
            "report_md": report_path,
        }
    return results, failures, artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CV research benchmark suite.")
    parser.add_argument("--projects", default="all", help="Comma-separated project ids to run, or 'all'.")
    parser.add_argument("--full", action="store_true", help="Run larger synthetic datasets instead of quick mode.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue when one project fails.")
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Optional root directory for generated reports and CSV artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = None if args.projects == "all" else [item.strip() for item in args.projects.split(",") if item.strip()]
    results, failures, artifacts = execute(
        quick=not args.full,
        selected_projects=selected,
        continue_on_error=args.continue_on_error,
        write_outputs=True,
        artifact_root=None if args.artifact_root is None else Path(args.artifact_root),
    )
    print(f"Completed {len(results)} project runs.")
    if failures:
        print("Failures:")
        for project, message in failures:
            print(f"- {project}: {message}")
    if artifacts:
        print("Artifacts:")
        for name, path in artifacts.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()