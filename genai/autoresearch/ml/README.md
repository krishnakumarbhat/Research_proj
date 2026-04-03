# GenAI Research Benchmark Suite

This folder contains a runnable CPU-first benchmark suite for the 15 small-data GenAI research directions requested in the workspace task.

## Run

From the `autoresearch` root:

```bash
uv run python -m ml.run_all
```

Run selected projects only:

```bash
uv run python -m ml.run_all --projects mechanistic_interpretability,black_box_prompt_optimization
```

Run the larger benchmark path:

```bash
uv run python -m ml.run_all --full --continue-on-error
```

## Outputs

- `research_report.md`: detailed markdown report with findings for every project
- `results/all_experiments.csv`: experiment-level table
- `results/project_summary.csv`: best result per project

Each project attempts to use the requested public dataset first and falls back to a synthetic proxy when the external dataset is unavailable or too heavy for the current environment.