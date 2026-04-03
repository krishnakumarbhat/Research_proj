from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "multimodal_results"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
DATA_CACHE = ROOT / "multimodal_data"
RANDOM_STATE = 42


@dataclass(slots=True)
class ExperimentRecord:
    project: str
    dataset: str
    source: str
    task: str
    algorithm: str
    feature_variant: str
    optimization: str
    primary_metric: str
    primary_value: float
    rank_score: float
    secondary_metric: str | None = None
    secondary_value: float | None = None
    tertiary_metric: str | None = None
    tertiary_value: float | None = None
    fit_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProjectResult:
    project: str
    title: str
    dataset: str
    source: str
    task: str
    records: list[ExperimentRecord]
    summary: str
    recommendation: str
    key_findings: list[str]
    caveats: list[str]
    artifacts: list[str]


def ensure_directories() -> None:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def timed_run(fn: Callable[[], object]) -> tuple[object, float]:
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def make_record(
    project: str,
    dataset: str,
    source: str,
    task: str,
    algorithm: str,
    feature_variant: str,
    optimization: str,
    primary_metric: str,
    primary_value: float,
    rank_score: float,
    fit_seconds: float,
    notes: str,
    secondary_metric: str | None = None,
    secondary_value: float | None = None,
    tertiary_metric: str | None = None,
    tertiary_value: float | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        project=project,
        dataset=dataset,
        source=source,
        task=task,
        algorithm=algorithm,
        feature_variant=feature_variant,
        optimization=optimization,
        primary_metric=primary_metric,
        primary_value=float(primary_value),
        rank_score=float(rank_score),
        secondary_metric=secondary_metric,
        secondary_value=None if secondary_value is None else float(secondary_value),
        tertiary_metric=tertiary_metric,
        tertiary_value=None if tertiary_value is None else float(tertiary_value),
        fit_seconds=float(fit_seconds),
        notes=notes,
    )


def records_to_frame(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    return pd.DataFrame(record.to_dict() for record in records)


def choose_best_record(records: Iterable[ExperimentRecord]) -> ExperimentRecord:
    return max(records, key=lambda record: record.rank_score)
