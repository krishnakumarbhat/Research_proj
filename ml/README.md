# ML Research Benchmark Suite

This folder contains a CPU-first research benchmark suite that turns the user's topic list into runnable Python experiments.

## Structure

- `run_all.py`: central runner that executes the selected projects, records all experiments, and writes CSV + Markdown artifacts.
- `run_all_project_files.py`: executes every `project_*.py` file as an entrypoint and records pass/fail status.
- `project_*.py`: one runnable experiment module per research topic.
- `results/`: generated CSV outputs.
- `research_report.md`: generated detailed report.
- `literature_review.md`: publish-oriented literature review draft aligned with the benchmark topics.
- `tests/test_smoke.py`: smoke tests for registry and quick-run execution.

## Run

From the repository root:

```bash
/usr/bin/python3 -m ml.run_all
```

Run selected projects only:

```bash
/usr/bin/python3 -m ml.run_all --projects ebm_credit,tree_noise_robustness,label_encoding_stability
```

Run each project file directly and record the results:

```bash
/usr/bin/python3 -m ml.run_all_project_files
```

Run the larger path:

```bash
/usr/bin/python3 -m ml.run_all --full --continue-on-error
```

## Notes

- Kaggle-gated datasets are supported through local file detection, but each runner also includes a documented fallback so the suite remains executable in a clean workspace.
- The generated Markdown report highlights the best model per project and records the individual experiment rows behind each recommendation.
- The strongest portfolio topic for minimal data remains the EBM credit-risk module, which is why the report calls it out explicitly.
