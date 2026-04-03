from __future__ import annotations

import argparse
from pathlib import Path

from ml.common import ProjectResult, ROOT, ensure_directories
from ml.project_agentic_data_validation import run as run_agentic_data_validation
from ml.project_causal_discovery import run as run_causal_discovery
from ml.project_concept_drift import run as run_concept_drift
from ml.project_conformal_supply_chain import run as run_conformal_supply_chain
from ml.project_ebm_credit import run as run_ebm_credit
from ml.project_evolutionary_feature_engineering import run as run_evolutionary_feature_engineering
from ml.project_federated_low_bandwidth import run as run_federated_low_bandwidth
from ml.project_graph_imbalance import run as run_graph_imbalance
from ml.project_label_encoding_stability import run as run_label_encoding_stability
from ml.project_memory_safe_feature_pipeline import run as run_memory_safe_feature_pipeline
from ml.project_security_aware_intrusion import run as run_security_aware_intrusion
from ml.project_synthetic_privacy import run as run_synthetic_privacy
from ml.project_tabular_meta_learning import run as run_tabular_meta_learning
from ml.project_tree_ensemble_optimization import run as run_tree_ensemble_optimization
from ml.project_tree_noise_robustness import run as run_tree_noise_robustness
from ml.reporting import write_csv_outputs, write_markdown_report


PROJECT_RUNNERS = {
    "causal_discovery": run_causal_discovery,
    "conformal_supply_chain": run_conformal_supply_chain,
    "ebm_credit": run_ebm_credit,
    "concept_drift": run_concept_drift,
    "federated_low_bandwidth": run_federated_low_bandwidth,
    "tabular_meta_learning": run_tabular_meta_learning,
    "synthetic_privacy": run_synthetic_privacy,
    "evolutionary_feature_engineering": run_evolutionary_feature_engineering,
    "graph_imbalance": run_graph_imbalance,
    "tree_noise_robustness": run_tree_noise_robustness,
    "memory_safe_feature_pipeline": run_memory_safe_feature_pipeline,
    "tree_ensemble_optimization": run_tree_ensemble_optimization,
    "label_encoding_stability": run_label_encoding_stability,
    "security_aware_intrusion": run_security_aware_intrusion,
    "agentic_data_validation": run_agentic_data_validation,
}


def execute(
    *,
    quick: bool = True,
    selected_projects: list[str] | None = None,
    continue_on_error: bool = False,
    write_outputs: bool = True,
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
            result = PROJECT_RUNNERS[project](quick=quick)
            results.append(result)
        except Exception as exc:
            failures.append((project, str(exc)))
            if not continue_on_error:
                raise

    artifacts: dict[str, Path] = {}
    if write_outputs and results:
        records_path, summary_path = write_csv_outputs(results, ROOT / "results")
        report_path = write_markdown_report(results, ROOT / "research_report.md")
        artifacts = {
            "records_csv": records_path,
            "summary_csv": summary_path,
            "report_md": report_path,
        }
    return results, failures, artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small-data ML research benchmark suite.")
    parser.add_argument(
        "--projects",
        default="all",
        help="Comma-separated project ids to run, or 'all'.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the larger benchmark path instead of quick mode.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining projects when one project fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = None if args.projects == "all" else [project.strip() for project in args.projects.split(",") if project.strip()]
    results, failures, artifacts = execute(
        quick=not args.full,
        selected_projects=selected,
        continue_on_error=args.continue_on_error,
        write_outputs=True,
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
