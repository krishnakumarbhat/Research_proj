from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .common import RESULTS_DIR


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    project_files = sorted(path for path in Path(__file__).resolve().parent.glob("project_*.py") if path.is_file())
    rows = ["# Project File Entrypoint Runs", "", "| File | Status | Runtime (s) | Notes |", "| --- | --- | --- | --- |"]
    passed = 0
    for path in project_files:
        start = perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", f"ml.{path.stem}"],
            cwd=path.parent.parent,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        runtime = perf_counter() - start
        if completed.returncode == 0:
            status = "PASS"
            notes = ""
            passed += 1
        else:
            status = "FAIL"
            notes = (completed.stderr or completed.stdout).splitlines()[0] if (completed.stderr or completed.stdout) else "non-zero exit"
        rows.append(f"| {path.name} | {status} | {runtime:.2f} | {notes.replace('|', '/')} |")

    artifact = RESULTS_DIR / "project_file_runs.md"
    artifact.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Completed {len(project_files)} project file runs; {passed} passed.")
    print(f"Artifact: {artifact}")


if __name__ == "__main__":
    main()