from __future__ import annotations

from typing import Iterable


def _stringify(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    materialized_rows = [[_stringify(cell) for cell in row] for row in rows]
    header_row = [_stringify(header) for header in headers]
    widths = [len(header) for header in header_row]
    for row in materialized_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        padded = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
        return "| " + " | ".join(padded) + " |"

    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(header_row), divider]
    lines.extend(format_row(row) for row in materialized_rows)
    return "\n".join(lines)
