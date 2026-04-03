# CPU-First RL Research Benchmark

This workspace contains a lightweight reinforcement-learning research suite focused on the 15 topics requested for the project. The code is built to run on a standard CPU using small tabular or simulator-lite environments, while still preserving the core research question behind each topic.

The benchmark lives in [rlbench/run_all.py](rlbench/run_all.py) and writes aggregated outputs to [results](results). A separate vendored copy of Karpathy's `autoresearch` project is already present in this workspace under [autoresearch](autoresearch), but it targets GPU LLM training and is intentionally kept separate from this RL benchmark.

## Quick Start

```bash
python3 -m rlbench.run_all
python3 -m pytest tests/test_smoke.py
```

## Outputs

- [results/all_experiments.csv](results/all_experiments.csv)
- [results/project_summary.csv](results/project_summary.csv)
- [research_report.md](research_report.md)
- [results/project_file_runs.md](results/project_file_runs.md)
- [results/test_results.md](results/test_results.md)

## Design Notes

- Each project exposes a `run(quick: bool = True)` entrypoint.
- Public datasets and simulator names from the original topic list are preserved in metadata.
- If those assets are not available locally, the runner falls back to synthetic but task-faithful generators.
- The suite compares multiple algorithms, feature/state variants, and optimization strategies per topic.