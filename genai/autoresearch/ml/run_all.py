from __future__ import annotations

import argparse
from pathlib import Path

from .common import ProjectArtifact, ROOT, ensure_directories
from .project_agentic_frameworks_hardware_synthesis import run as run_agentic_frameworks_hardware_synthesis
from .project_black_box_prompt_optimization import run as run_black_box_prompt_optimization
from .project_constrained_generation_logit_masking import run as run_constrained_generation_logit_masking
from .project_context_window_compression import run as run_context_window_compression
from .project_direct_preference_optimization import run as run_direct_preference_optimization
from .project_edge_optimized_rag import run as run_edge_optimized_rag
from .project_hallucination_detection import run as run_hallucination_detection
from .project_mechanistic_interpretability import run as run_mechanistic_interpretability
from .project_qlora_sub_billion_legal import run as run_qlora_sub_billion_legal
from .project_rag_chunking_optimization import run as run_rag_chunking_optimization
from .project_small_ai_agent_workflows import run as run_small_ai_agent_workflows
from .project_small_model_orchestration_ci_cd import run as run_small_model_orchestration_ci_cd
from .project_stateful_multi_agent_debugging import run as run_stateful_multi_agent_debugging
from .project_structural_bias_evaluation import run as run_structural_bias_evaluation
from .project_synthetic_data_generation_niche_classifiers import run as run_synthetic_data_generation_niche_classifiers
from .reporting import write_csv_outputs, write_markdown_report


PROJECT_RUNNERS = {
    "stateful_multi_agent_debugging": run_stateful_multi_agent_debugging,
    "constrained_generation_logit_masking": run_constrained_generation_logit_masking,
    "edge_optimized_rag": run_edge_optimized_rag,
    "agentic_frameworks_hardware_synthesis": run_agentic_frameworks_hardware_synthesis,
    "small_model_orchestration_ci_cd": run_small_model_orchestration_ci_cd,
    "mechanistic_interpretability": run_mechanistic_interpretability,
    "rag_chunking_optimization": run_rag_chunking_optimization,
    "small_ai_agent_workflows": run_small_ai_agent_workflows,
    "qlora_sub_billion_legal": run_qlora_sub_billion_legal,
    "black_box_prompt_optimization": run_black_box_prompt_optimization,
    "hallucination_detection": run_hallucination_detection,
    "context_window_compression": run_context_window_compression,
    "direct_preference_optimization": run_direct_preference_optimization,
    "synthetic_data_generation_niche_classifiers": run_synthetic_data_generation_niche_classifiers,
    "structural_bias_evaluation": run_structural_bias_evaluation,
}


def execute(
    *,
    quick: bool = True,
    selected_projects: list[str] | None = None,
    continue_on_error: bool = False,
    write_outputs: bool = True,
) -> tuple[list[ProjectArtifact], list[tuple[str, str]], dict[str, Path]]:
    ensure_directories()
    selected = selected_projects or list(PROJECT_RUNNERS.keys())
    results: list[ProjectArtifact] = []
    failures: list[tuple[str, str]] = []
    for project_id in selected:
        if project_id not in PROJECT_RUNNERS:
            message = f"Unknown project id: {project_id}"
            if continue_on_error:
                failures.append((project_id, message))
                continue
            raise KeyError(message)
        try:
            results.append(PROJECT_RUNNERS[project_id](quick=quick))
        except Exception as exc:
            failures.append((project_id, str(exc)))
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
    parser = argparse.ArgumentParser(description="Run the GenAI small-data research benchmark suite.")
    parser.add_argument("--projects", default="all", help="Comma-separated project ids to run, or 'all'.")
    parser.add_argument("--full", action="store_true", help="Run the larger benchmark path instead of quick mode.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining projects when one project fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = None if args.projects == "all" else [part.strip() for part in args.projects.split(",") if part.strip()]
    results, failures, artifacts = execute(
        quick=not args.full,
        selected_projects=selected,
        continue_on_error=args.continue_on_error,
        write_outputs=True,
    )
    print(f"Completed {len(results)} project runs.")
    if failures:
        print("Failures:")
        for project_id, message in failures:
            print(f"- {project_id}: {message}")
    if artifacts:
        print("Artifacts:")
        for name, path in artifacts.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()