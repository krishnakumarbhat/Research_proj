from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from rlbench.common import ProjectResult, ROOT, choose_best_record, records_to_frame


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_row, divider_row, *body_rows])


def write_csv_outputs(project_results: Iterable[ProjectResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = [record for result in project_results for record in result.records]
    records_path = output_dir / "all_experiments.csv"
    records_to_frame(all_records).to_csv(records_path, index=False)

    summary_rows = []
    for result in project_results:
        best = choose_best_record(result.records)
        summary_rows.append(
            {
                "project": result.project,
                "title": result.title,
                "dataset": result.dataset,
                "best_algorithm": best.algorithm,
                "primary_metric": best.primary_metric,
                "primary_value": best.primary_value,
                "recommendation": result.recommendation,
            }
        )
    summary_path = output_dir / "project_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return records_path, summary_path


def write_markdown_report(project_results: list[ProjectResult], output_path: Path) -> Path:
    rows = []
    for result in project_results:
        best = choose_best_record(result.records)
        rows.append(
            [
                result.title,
                result.dataset,
                best.algorithm,
                f"{best.primary_metric}={best.primary_value:.4f}",
                result.recommendation,
            ]
        )

    sections = [
        "# CPU-First RL Research Benchmark Report",
        "",
        "This report summarizes 15 lightweight reinforcement-learning research projects implemented in `rlbench/`. The benchmark keeps Karpathy's vendored `autoresearch/` project untouched and instead borrows its experiment-loop mindset: compare compact variants quickly, keep explicit logs, and prioritize reproducible CPU runs.",
        "",
        "Heavy external simulators and Kaggle-gated assets are represented by local-data hooks plus synthetic fallbacks. That keeps every project runnable in this workspace while preserving the main optimization problem behind the original topic.",
        "",
        "## Cross-Project Summary",
        markdown_table(["Project", "Dataset", "Best Model", "Best Score", "Recommendation"], rows),
    ]

    for result in project_results:
        best = choose_best_record(result.records)
        project_rows = []
        for record in sorted(result.records, key=lambda item: item.rank_score, reverse=True):
            secondary = (
                f"{record.secondary_metric}={record.secondary_value:.4f}"
                if record.secondary_metric and record.secondary_value is not None
                else ""
            )
            tertiary = (
                f"{record.tertiary_metric}={record.tertiary_value:.4f}"
                if record.tertiary_metric and record.tertiary_value is not None
                else ""
            )
            project_rows.append(
                [
                    record.algorithm,
                    record.feature_variant,
                    record.optimization,
                    f"{record.primary_metric}={record.primary_value:.4f}",
                    secondary,
                    tertiary,
                    f"{record.fit_seconds:.2f}s",
                    record.notes,
                ]
            )
        sections.extend(
            [
                "",
                f"## {result.title}",
                "",
                f"**Dataset:** {result.dataset}",
                "",
                f"**Best experiment:** {best.algorithm} with {best.primary_metric}={best.primary_value:.4f}",
                "",
                result.summary,
                "",
                "### Key Findings",
                *[f"- {finding}" for finding in result.key_findings],
                "",
                "### Recorded Experiments",
                markdown_table(
                    ["Algorithm", "Features", "Optimization", "Primary", "Secondary", "Tertiary", "Runtime", "Notes"],
                    project_rows,
                ),
                "",
                "### Caveats",
                *[f"- {caveat}" for caveat in result.caveats],
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return output_path


def report_paths() -> dict[str, Path]:
    return {
        "report_md": ROOT / "research_report.md",
        "records_csv": ROOT / "results" / "all_experiments.csv",
        "summary_csv": ROOT / "results" / "project_summary.csv",
    }