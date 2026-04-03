from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable


def _escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _tick_values(minimum: float, maximum: float, steps: int = 5) -> list[float]:
    if steps <= 1:
        return [minimum]
    if maximum <= minimum:
        maximum = minimum + 1.0
    step = (maximum - minimum) / (steps - 1)
    return [minimum + (index * step) for index in range(steps)]


def horizontal_bar_chart_svg(
    *,
    title: str,
    items: Iterable[dict[str, object]],
    x_label: str,
    width: int = 960,
) -> str:
    rows = list(items)
    row_height = 30
    top = 72
    bottom = 64
    left = 280
    right = 96
    bar_height = 18
    plot_width = width - left - right
    height = top + bottom + (len(rows) * row_height)
    maximum = max((float(row["value"]) for row in rows), default=1.0)
    if maximum <= 0.0:
        maximum = 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        "<style>",
        ".title { font: 700 22px sans-serif; fill: #172033; }",
        ".axis { font: 12px sans-serif; fill: #4b5563; }",
        ".label { font: 13px sans-serif; fill: #172033; }",
        ".value { font: 12px sans-serif; fill: #172033; }",
        ".grid { stroke: #d6dce7; stroke-width: 1; }",
        ".baseline { stroke: #5b6472; stroke-width: 1.5; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="28" y="34">{_escape(title)}</text>',
    ]

    for tick in _tick_values(0.0, maximum):
        x = left + (plot_width * tick / maximum)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top - 8}" x2="{x:.2f}" y2="{height - bottom + 8}"/>')
        parts.append(f'<text class="axis" x="{x:.2f}" y="{height - bottom + 28}" text-anchor="middle">{tick:.3f}</text>')

    parts.append(f'<line class="baseline" x1="{left}" y1="{top - 8}" x2="{left}" y2="{height - bottom + 8}"/>')
    parts.append(f'<text class="axis" x="{left + (plot_width / 2):.2f}" y="{height - 18}" text-anchor="middle">{_escape(x_label)}</text>')

    for index, row in enumerate(rows):
        y = top + (index * row_height)
        label_y = y + bar_height - 2
        value = float(row["value"])
        bar_width = 0.0 if value <= 0.0 else plot_width * value / maximum
        color = str(row.get("color", "#4c78a8"))
        parts.append(f'<text class="label" x="{left - 12}" y="{label_y:.2f}" text-anchor="end">{_escape(row["label"])}</text>')
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="{_escape(color)}" rx="4" ry="4"/>'
        )
        parts.append(
            f'<text class="value" x="{left + bar_width + 8:.2f}" y="{label_y:.2f}">{value:.4f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def scatter_chart_svg(
    *,
    title: str,
    items: Iterable[dict[str, object]],
    x_label: str,
    y_label: str,
    width: int = 960,
    height: int = 620,
) -> str:
    rows = list(items)
    left = 88
    right = 180
    top = 72
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row["x"]) for row in rows] or [0.0, 1.0]
    y_values = [float(row["y"]) for row in rows] or [0.0, 1.0]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    x_min -= x_span * 0.05
    x_max += x_span * 0.08
    y_min -= y_span * 0.08
    y_max += y_span * 0.10

    def map_x(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min) * plot_width)

    def map_y(value: float) -> float:
        return top + plot_height - ((value - y_min) / (y_max - y_min) * plot_height)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        "<style>",
        ".title { font: 700 22px sans-serif; fill: #172033; }",
        ".axis { font: 12px sans-serif; fill: #4b5563; }",
        ".label { font: 12px sans-serif; fill: #172033; }",
        ".grid { stroke: #d6dce7; stroke-width: 1; }",
        ".baseline { stroke: #5b6472; stroke-width: 1.5; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="28" y="34">{_escape(title)}</text>',
    ]

    for tick in _tick_values(x_min, x_max):
        x = map_x(tick)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        parts.append(f'<text class="axis" x="{x:.2f}" y="{height - bottom + 28}" text-anchor="middle">{tick:.2f}</text>')

    for tick in _tick_values(y_min, y_max):
        y = map_y(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        parts.append(f'<text class="axis" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.3f}</text>')

    parts.append(f'<line class="baseline" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="baseline" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<text class="axis" x="{left + (plot_width / 2):.2f}" y="{height - 20}" text-anchor="middle">{_escape(x_label)}</text>')
    parts.append(
        f'<text class="axis" x="26" y="{top + (plot_height / 2):.2f}" text-anchor="middle" transform="rotate(-90 26 {top + (plot_height / 2):.2f})">{_escape(y_label)}</text>'
    )

    for row in rows:
        x = map_x(float(row["x"]))
        y = map_y(float(row["y"]))
        color = str(row.get("color", "#4c78a8"))
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{_escape(color)}" opacity="0.9"/>')
        parts.append(f'<text class="label" x="{x + 8:.2f}" y="{y - 8:.2f}">{_escape(row["label"])}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination