from __future__ import annotations

import argparse
from pathlib import Path

from dl.common import ProjectResult, ROOT, ensure_directories
from dl.project_adversarial_edge_robustness import run as run_adversarial_edge_robustness
from dl.project_audio_autoencoder import run as run_audio_autoencoder
from dl.project_biosignal_diagnostics import run as run_biosignal_diagnostics
from dl.project_continual_hashing import run as run_continual_hashing
from dl.project_continuous_edge_learning import run as run_continuous_edge_learning
from dl.project_deterministic_bnn import run as run_deterministic_bnn
from dl.project_hardware_aware_nas import run as run_hardware_aware_nas
from dl.project_knowledge_distillation_api import run as run_knowledge_distillation_api
from dl.project_liquid_financial_signals import run as run_liquid_financial_signals
from dl.project_physics_informed_pinn import run as run_physics_informed_pinn
from dl.project_quantization_time_series import run as run_quantization_time_series
from dl.project_sensor_fusion_architectures import run as run_sensor_fusion_architectures
from dl.project_spiking_edge_anomaly import run as run_spiking_edge_anomaly
from dl.project_tinyml_micro_speech import run as run_tinyml_micro_speech
from dl.project_weight_quantization_theory import run as run_weight_quantization_theory
from dl.reporting import write_csv_outputs, write_markdown_report


PROJECT_RUNNERS = {
    "sensor_fusion_architectures": run_sensor_fusion_architectures,
    "hardware_aware_nas": run_hardware_aware_nas,
    "quantization_time_series": run_quantization_time_series,
    "deterministic_bnn": run_deterministic_bnn,
    "continuous_edge_learning": run_continuous_edge_learning,
    "tinyml_micro_speech": run_tinyml_micro_speech,
    "liquid_financial_signals": run_liquid_financial_signals,
    "biosignal_diagnostics": run_biosignal_diagnostics,
    "knowledge_distillation_api": run_knowledge_distillation_api,
    "spiking_edge_anomaly": run_spiking_edge_anomaly,
    "physics_informed_pinn": run_physics_informed_pinn,
    "audio_autoencoder": run_audio_autoencoder,
    "weight_quantization_theory": run_weight_quantization_theory,
    "continual_hashing": run_continual_hashing,
    "adversarial_edge_robustness": run_adversarial_edge_robustness,
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
    parser = argparse.ArgumentParser(description="Run the CPU-first deep learning benchmark suite.")
    parser.add_argument("--projects", default="all", help="Comma-separated project ids to run, or 'all'.")
    parser.add_argument("--full", action="store_true", help="Run the larger benchmark path instead of quick mode.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running remaining projects when one project fails.")
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
