from research_suite.common.charts import horizontal_bar_chart_svg, scatter_chart_svg


def test_horizontal_bar_chart_svg_contains_labels_and_title() -> None:
    svg = horizontal_bar_chart_svg(
        title="Example Bar Chart",
        items=[
            {"label": "alpha", "value": 0.1, "color": "#111111"},
            {"label": "beta", "value": 0.3, "color": "#222222"},
        ],
        x_label="score",
    )
    assert svg.startswith("<svg")
    assert "Example Bar Chart" in svg
    assert "alpha" in svg
    assert "beta" in svg


def test_scatter_chart_svg_contains_points_and_title() -> None:
    svg = scatter_chart_svg(
        title="Example Scatter Chart",
        items=[
            {"label": "point-a", "x": 1.0, "y": 0.2, "color": "#111111"},
            {"label": "point-b", "x": 2.0, "y": 0.4, "color": "#222222"},
        ],
        x_label="latency",
        y_label="accuracy",
    )
    assert svg.startswith("<svg")
    assert "Example Scatter Chart" in svg
    assert "point-a" in svg
    assert "point-b" in svg