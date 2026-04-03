from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT / "projects"
RESULTS_DIR = ROOT / "results"


@dataclass(slots=True)
class ProjectFileRun:
    file_name: str
    status: str
    runtime_seconds: float
    summary_line: str
    details: str = ""


def _discover_project_files() -> list[Path]:
    return sorted(path for path in PROJECTS_DIR.glob("project_*.py") if path.is_file())


def _summarize_stderr(stderr: str, module_name: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return module_name
    return f"{module_name}; emitted {len(lines)} stderr lines"


def _run_file(path: Path, timeout: int = 300) -> ProjectFileRun:
    start = perf_counter()
    module_name = f"cv_bench.projects.{path.stem}"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        runtime = perf_counter() - start
        summary = (completed.stdout.strip().splitlines() or [""])[0]
        return ProjectFileRun(
            file_name=path.name,
            status="PASS" if completed.returncode == 0 else f"FAIL ({completed.returncode})",
            runtime_seconds=runtime,
            summary_line=summary,
            details=_summarize_stderr(completed.stderr, module_name),
        )
    except subprocess.TimeoutExpired:
        runtime = perf_counter() - start
        return ProjectFileRun(
            file_name=path.name,
            status="TIMEOUT",
            runtime_seconds=runtime,
            summary_line="",
            details=f"{module_name}; timed out after {timeout} seconds",
        )


def _write_markdown(results: list[ProjectFileRun], output_path: Path) -> Path:
    rows = [
        "# Project File Entrypoint Runs",
        "",
        "| File | Status | Runtime (s) | Summary | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        rows.append(
            f"| {result.file_name} | {result.status} | {result.runtime_seconds:.2f} | {result.summary_line or 'n/a'} | {(result.details or '').replace('|', '/')} |"
        )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = [_run_file(path) for path in _discover_project_files()]
    output_path = _write_markdown(results, RESULTS_DIR / "project_file_runs.md")
    passed = sum(result.status == "PASS" for result in results)
    print(f"Completed {len(results)} project file runs; {passed} passed.")
    print(f"Artifact: {output_path}")


if __name__ == "__main__":
    main()