from __future__ import annotations

import argparse
from pathlib import Path

from research_suite.common.io_utils import ensure_directory
from research_suite.common.reporting import (
    build_detailed_report,
    build_summary_report,
    generate_chart_bundle,
    load_artifact_results,
)
from research_suite.paper1_visual_text.experiment import run as run_paper1
from research_suite.paper2_dynamic_rag.experiment import run as run_paper2
from research_suite.paper3_chunking.experiment import run as run_paper3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research experiments for the three paper prototypes.")
    parser.add_argument("--quick", action="store_true", help="Run smaller datasets for fast iteration.")
    parser.add_argument(
        "--paper",
        choices=["all", "paper1", "paper2", "paper3"],
        default="all",
        help="Select a single paper experiment or run all of them.",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Rebuild reports and charts from the current artifact CSV and JSON files without rerunning experiments.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent
    artifacts_dir = ensure_directory(workspace_root / "artifacts")
    reports_dir = ensure_directory(workspace_root / "reports")
    if args.reports_only:
        results = load_artifact_results(artifacts_dir)
        chart_paths = generate_chart_bundle(results=results, artifacts_dir=artifacts_dir)
        build_summary_report(results=results, reports_dir=reports_dir)
        build_detailed_report(results=results, reports_dir=reports_dir, chart_paths=chart_paths)
        return

    results: dict[str, dict[str, object]] = {}

    if args.paper in {"all", "paper1"}:
        results["paper1"] = run_paper1(output_root=artifacts_dir, quick=args.quick)
    if args.paper in {"all", "paper2"}:
        results["paper2"] = run_paper2(output_root=artifacts_dir, quick=args.quick)
    if args.paper in {"all", "paper3"}:
        results["paper3"] = run_paper3(output_root=artifacts_dir, quick=args.quick)

    if args.paper == "all":
        chart_paths = generate_chart_bundle(results=results, artifacts_dir=artifacts_dir)
        build_summary_report(results=results, reports_dir=reports_dir)
        build_detailed_report(results=results, reports_dir=reports_dir, chart_paths=chart_paths)


if __name__ == "__main__":
    main()
