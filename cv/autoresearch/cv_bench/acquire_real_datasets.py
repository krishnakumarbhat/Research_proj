from __future__ import annotations

import argparse
from pathlib import Path

from cv_bench.common import ROOT
from cv_bench.real_data import download_supported_real_datasets


def _write_status_markdown(statuses: list[dict[str, str]], output_path: Path) -> Path:
    rows = [
        "# Real Dataset Acquisition",
        "",
        "| Dataset | Status | Path | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for entry in statuses:
        rows.append(
            f"| {entry['dataset']} | {entry['status']} | {entry['path']} | {entry['notes']} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download small public real-dataset subsets for the CV benchmark.")
    parser.add_argument(
        "--skip-nmnist",
        action="store_true",
        help="Skip the larger N-MNIST archive download.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    statuses = download_supported_real_datasets(include_n_mnist=not args.skip_nmnist)
    output_path = _write_status_markdown(statuses, ROOT / "results" / "real_dataset_status.md")
    for entry in statuses:
        print(f"{entry['dataset']}: {entry['status']} -> {entry['path']}")
    print(f"Artifact: {output_path}")


if __name__ == "__main__":
    main()