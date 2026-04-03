from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from cv_bench.common import ProjectResult, choose_best_record, records_to_frame


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
                "source": best.source,
            }
        )
    summary_path = output_dir / "project_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return records_path, summary_path


def write_markdown_report(project_results: list[ProjectResult], output_path: Path) -> Path:
    summary_rows = []
    for result in project_results:
        best = choose_best_record(result.records)
        summary_rows.append(
            [
                result.title,
                result.dataset,
                best.source,
                best.algorithm,
                f"{best.primary_metric}={best.primary_value:.4f}",
                result.recommendation,
            ]
        )

    sections = [
        "# CV Research Benchmark Report",
        "",
        "This report was generated from runnable experiment modules under `cv_bench/`. Each project benchmark compares multiple algorithms, feature variants, and optimization choices. When a requested dataset is not available locally, the suite falls back to a deterministic synthetic generator so the code, tests, and reporting pipeline remain executable.",
        "",
        "## Cross-Project Summary",
        markdown_table(
            ["Project", "Dataset", "Source", "Best Model", "Best Score", "Recommendation"],
            summary_rows,
        ),
    ]

    for result in project_results:
        best = choose_best_record(result.records)
        project_rows = []
        for record in sorted(result.records, key=lambda item: item.rank_score, reverse=True):
            secondary = ""
            if record.secondary_metric and record.secondary_value is not None:
                secondary = f"{record.secondary_metric}={record.secondary_value:.4f}"
            tertiary = ""
            if record.tertiary_metric and record.tertiary_value is not None:
                tertiary = f"{record.tertiary_metric}={record.tertiary_value:.4f}"
            project_rows.append(
                [
                    record.algorithm,
                    record.feature_variant,
                    record.optimization,
                    record.source,
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
                    [
                        "Algorithm",
                        "Features",
                        "Optimization",
                        "Source",
                        "Primary",
                        "Secondary",
                        "Tertiary",
                        "Runtime",
                        "Notes",
                    ],
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