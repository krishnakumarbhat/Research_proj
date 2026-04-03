# RL Benchmark Test Results

## Smoke Tests

- Command: `python3 -m unittest tests.test_smoke`
- Result: `OK`
- Tests run: `2`

## Entrypoint Runs

- Command: `python3 -m rlbench.run_all_project_files`
- Result: `15/15` project module entrypoints passed

## Benchmark Run

- Command: `python3 -m rlbench.run_all --continue-on-error`
- Result: `15/15` project benchmarks completed successfully
- Artifacts generated:
  - `results/all_experiments.csv`
  - `results/project_summary.csv`
  - `results/project_file_runs.md`
  - `research_report.md`