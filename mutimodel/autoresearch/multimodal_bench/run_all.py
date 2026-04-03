from __future__ import annotations

import argparse
from pathlib import Path

from .asr_digits import run as run_asr
from .common import RESULTS_DIR, ensure_directories
from .ocr_digits import run as run_ocr
from .reporting import write_csv_outputs, write_markdown_report
from .text_to_image_digits import run as run_text_to_image
from .text_to_speech_digits import run as run_text_to_speech


def execute(quick: bool = False, prefer_real_audio: bool = True) -> tuple[list, dict[str, Path]]:
    ensure_directories()
    project_results = []

    asr_result, asr_model = run_asr(quick=quick, prefer_real_audio=prefer_real_audio)
    project_results.append(asr_result)

    ocr_result, ocr_model = run_ocr(quick=quick)
    project_results.append(ocr_result)

    project_results.append(run_text_to_image(ocr_model=ocr_model, quick=quick))
    project_results.append(
        run_text_to_speech(asr_model=asr_model, quick=quick, prefer_real_audio=prefer_real_audio)
    )

    records_path, summary_path = write_csv_outputs(project_results, RESULTS_DIR)
    report_path = write_markdown_report(project_results, RESULTS_DIR / "multimodal_report.md")
    return project_results, {
        "records_csv": records_path,
        "summary_csv": summary_path,
        "report_md": report_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multimodal mini-benchmark suite.")
    parser.add_argument("--quick", action="store_true", help="Use the reduced benchmark path.")
    parser.add_argument(
        "--force-synthetic-audio",
        action="store_true",
        help="Skip the real spoken-digit download and use synthetic audio instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_results, artifacts = execute(
        quick=args.quick,
        prefer_real_audio=not args.force_synthetic_audio,
    )
    print(f"Completed {len(project_results)} multimodal projects.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
