# Benchmark Validation

## Smoke Tests

| Command | Status | Tests | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `/usr/bin/python3 -m pytest ml/tests/test_smoke.py` | PASS | 2 | 76.92s | Registry and selected-project execution smoke tests passed after the variation expansion. |

## Project File Entrypoints

| Command | Status | Files | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `/usr/bin/python3 -m ml.run_all_project_files` | PASS | 15 | See `ml/results/project_file_runs.md` | Every `project_*.py` entrypoint completed successfully when executed as a package module. |

## Benchmark Execution Notes

| Run Type | Status | Notes |
| --- | --- | --- |
| Quick all-project benchmark | COMPLETE | 15 projects completed and 112 experiment rows were recorded into the current report and CSV artifacts. |
| Full non-quick benchmark | ATTEMPTED / STOPPED | Started with `/usr/bin/python3 -m ml.run_all --full --continue-on-error`, remained active for well over 25 minutes, and was stopped to avoid leaving an indefinite background process. No partial full-run artifacts were used in the report. |
