from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .common import ProjectResult, choose_best_record, records_to_frame


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_row, divider_row] + body_rows)


def write_csv_outputs(project_results: Iterable[ProjectResult], output_dir: Path) -> tuple[Path, Path]:
    all_records = [record for result in project_results for record in result.records]
    records_frame = records_to_frame(all_records)
    records_path = output_dir / "all_experiments.csv"
    records_frame.to_csv(records_path, index=False)

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
    sections: list[str] = []
    sections.append("# Multimodal AI Mini-Benchmark Report")
    sections.append("")
    sections.append(
        "This report was generated from runnable experiments inside the current workspace. "
        "The benchmarks are intentionally compact: spoken-digit ASR and TTS proxies, digit OCR, "
        "and conditional digit-image generation used as a text-to-image proxy."
    )
    sections.append("")
    sections.append("## Cross-Project Summary")

    summary_rows = []
    for result in project_results:
        best = choose_best_record(result.records)
        summary_rows.append(
            [
                result.title,
                result.dataset,
                best.algorithm,
                f"{best.primary_metric}={best.primary_value:.4f}",
                result.recommendation,
            ]
        )
    sections.append(
        markdown_table(
            ["Project", "Dataset", "Best Model", "Best Score", "Recommendation"],
            summary_rows,
        )
    )

    for result in project_results:
        best = choose_best_record(result.records)
        sections.append("")
        sections.append(f"## {result.title}")
        sections.append("")
        sections.append(f"**Dataset:** {result.dataset}")
        sections.append(f"**Source:** {result.source}")
        sections.append(f"**Best experiment:** {best.algorithm} with {best.primary_metric}={best.primary_value:.4f}")
        sections.append("")
        sections.append(result.summary)
        sections.append("")
        sections.append("### Recommendation")
        sections.append(result.recommendation)
        sections.append("")
        sections.append("### Key Findings")
        sections.extend(f"- {finding}" for finding in result.key_findings)
        sections.append("")
        sections.append("### Recorded Experiments")

        project_rows = []
        for record in result.records:
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
        sections.append(
            markdown_table(
                [
                    "Algorithm",
                    "Features",
                    "Optimization",
                    "Primary",
                    "Secondary",
                    "Tertiary",
                    "Runtime",
                    "Notes",
                ],
                project_rows,
            )
        )

        if result.artifacts:
            sections.append("")
            sections.append("### Artifacts")
            for artifact in result.artifacts:
                if artifact.endswith(".png"):
                    sections.append(f"![]({artifact})")
                else:
                    sections.append(f"- {artifact}")

        sections.append("")
        sections.append("### Caveats")
        sections.extend(f"- {caveat}" for caveat in result.caveats)

    output_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return output_path