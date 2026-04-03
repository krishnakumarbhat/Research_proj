from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .common import ProjectArtifact, markdown_table, pick_best_result


def write_csv_outputs(projects: Iterable[ProjectArtifact], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    project_list = list(projects)

    all_rows = [result.to_row() for project in project_list for result in project.results]
    all_path = output_dir / "all_experiments.csv"
    pd.DataFrame(all_rows).to_csv(all_path, index=False)

    summary_rows = []
    for project in project_list:
        best = pick_best_result(project.results)
        summary_rows.append(
            {
                "project_id": project.project_id,
                "title": project.title,
                "used_dataset": project.used_dataset,
                "best_variant": best.variant if best else "",
                "best_algorithm": best.algorithm if best else "",
                "metric_name": best.metric_name if best else project.metric_name,
                "metric_value": best.metric_value if best else None,
                "objective": project.objective,
            }
        )
    summary_path = output_dir / "project_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return all_path, summary_path


def write_markdown_report(projects: Iterable[ProjectArtifact], output_path: Path) -> Path:
    project_list = list(projects)
    summary_rows = []
    for project in project_list:
        best = pick_best_result(project.results)
        summary_rows.append(
            {
                "Project": project.title,
                "Dataset": project.used_dataset,
                "Best Variant": best.variant if best else "",
                "Best Score": f"{best.metric_value:.4f}" if best else "",
                "Metric": best.metric_name if best else project.metric_name,
            }
        )

    sections = [
        "# Small-Data GenAI Research Benchmark Report",
        "",
        "This report was generated from runnable lightweight experiment scripts under `ml/`. Each project benchmarks multiple algorithms, feature variants, or optimization strategies under CPU-first constraints.",
        "",
        "## Cross-Project Summary",
        "",
        markdown_table(summary_rows, ["Project", "Dataset", "Best Variant", "Best Score", "Metric"]),
    ]

    for project in project_list:
        best = pick_best_result(project.results)
        result_rows = []
        for result in project.results:
            result_rows.append(
                {
                    "Variant": result.variant,
                    "Algorithm": result.algorithm,
                    "Features": result.feature_set,
                    "Primary": f"{result.metric_name}={result.metric_value:.4f}",
                    "Secondary": (
                        f"{result.secondary_metric_name}={result.secondary_metric_value:.4f}"
                        if result.secondary_metric_name and result.secondary_metric_value is not None
                        else ""
                    ),
                    "Runtime(s)": f"{result.runtime_sec:.2f}",
                    "Notes": result.notes,
                }
            )

        sections.extend(
            [
                "",
                f"## {project.title}",
                "",
                f"Dataset requested: {project.requested_dataset}",
                f"Dataset used: {project.used_dataset}",
                f"Objective: {project.objective}",
                f"Best experiment: {best.variant} with {best.metric_name}={best.metric_value:.4f}" if best else "Best experiment: unavailable",
                "",
                "### Findings",
                *[f"- {finding}" for finding in project.findings],
                "",
                "### Recorded Experiments",
                "",
                markdown_table(result_rows, ["Variant", "Algorithm", "Features", "Primary", "Secondary", "Runtime(s)", "Notes"]),
            ]
        )
        if project.notes:
            sections.extend(["", "### Notes", "", project.notes])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return output_path