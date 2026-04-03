# Deep Learning Research Suite

This folder mirrors the `ml/` benchmark structure but focuses on compact deep-learning projects that still run end to end on CPU. Each project is a standalone `project_*.py` module with a `run(quick=True)` entry point, and the shared runner records experiment tables plus a Markdown report.

## Design Principles

- CPU-first: all benchmarks are sized to finish on a laptop-class CPU in quick mode.
- Karpathy-inspired iteration: short training loops, minimal abstractions, and reviewable experiments rather than opaque pipelines.
- Real datasets when feasible, synthetic fallbacks when remote assets are gated or impractical.
- Multiple variants per project: model family, feature path, and optimization choice are all tracked in the experiment ledger.

## Relation to `autoresearch/`

The repository already contains Karpathy's `autoresearch/` project locally. That code targets a single NVIDIA GPU and autonomous training of a GPT-style model. This `dl/` suite does not depend on it directly. Instead, it adopts the same rapid-experiment ethos for small-data deep learning tasks that are realistic on CPU.

## Structure

- `common.py`: shared records, metrics, trainers, and utility helpers.
- `data.py`: lightweight real/synthetic dataset loaders.
- `models.py`: reusable PyTorch models for MLPs, 1D CNNs, GRUs, liquid-style RNNs, and autoencoders.
- `run_all.py`: central benchmark runner that writes CSV and Markdown artifacts.
- `run_all_project_files.py`: executes each `project_*.py` module as a package entrypoint.
- `project_*.py`: one benchmark module per DL topic.
- `results/`: generated CSV and validation artifacts.
- `tests/test_smoke.py`: smoke tests.

## Commands

Run the quick DL benchmark suite:

```bash
/usr/bin/python3 -m dl.run_all --continue-on-error
```

Run selected DL projects only:

```bash
/usr/bin/python3 -m dl.run_all --projects sensor_fusion_architectures,quantization_time_series,physics_informed_pinn
```

Run every project module entrypoint directly:

```bash
/usr/bin/python3 -m dl.run_all_project_files
```
