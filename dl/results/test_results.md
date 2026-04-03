# DL Benchmark Validation

## Smoke Tests

| Command | Status | Tests | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `/usr/bin/python3 -m pytest dl/tests/test_smoke.py` | PASS | 2 | 124.53s | Registry coverage and selected-project quick execution passed for the DL suite. |

## Project File Entrypoints

| Command | Status | Files | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `/usr/bin/python3 -m dl.run_all_project_files` | PASS | 15 | See `dl/results/project_file_runs.md` | Every `project_*.py` DL entrypoint completed successfully when executed as a package module. |

## Benchmark Execution Notes

| Run Type | Status | Notes |
| --- | --- | --- |
| Quick all-project benchmark | COMPLETE | 15 projects completed and 57 experiment rows were recorded into the current DL report and CSV artifacts. |
| Full non-quick benchmark | NOT RUN | The quick benchmark was completed end to end first because the user stopped a longer validation run. Full-mode execution can be batched later if needed. |
