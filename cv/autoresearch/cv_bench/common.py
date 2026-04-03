from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_CACHE = ROOT / "data_cache"
RESULTS_DIR = ROOT / "results"


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


@dataclass(slots=True)
class ProjectResult:
    project: str
    title: str
    dataset: str
    records: list[ExperimentRecord]
    summary: str
    recommendation: str
    key_findings: list[str]
    caveats: list[str]


def ensure_directories() -> None:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timed_run(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def configured_n_jobs(default: int = 1) -> int:
    raw_value = os.environ.get("CV_BENCH_N_JOBS")
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(1, value)


def make_record(
    *,
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
    secondary_metric: str | None = None,
    secondary_value: float | None = None,
    tertiary_metric: str | None = None,
    tertiary_value: float | None = None,
    fit_seconds: float = 0.0,
    notes: str = "",
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


def choose_best_record(records: Sequence[ExperimentRecord]) -> ExperimentRecord:
    if not records:
        raise ValueError("No experiment records were produced")
    return max(records, key=lambda record: record.rank_score)


def records_to_frame(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    return pd.DataFrame(asdict(record) for record in records)


def maybe_downsample(frame: pd.DataFrame, max_rows: int, stratify: str | None = None) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    if stratify is None:
        return frame.sample(max_rows, random_state=42).reset_index(drop=True)

    sampled = []
    fractions = frame[stratify].value_counts(normalize=True)
    for label, fraction in fractions.items():
        group = frame.loc[frame[stratify] == label]
        take = max(1, int(round(max_rows * fraction)))
        sampled.append(group.sample(min(len(group), take), random_state=42))
    return pd.concat(sampled, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)


def one_hot_align(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.get_dummies(train_frame, dummy_na=True)
    test_encoded = pd.get_dummies(test_frame, dummy_na=True)
    train_aligned, test_aligned = train_encoded.align(test_encoded, join="outer", axis=1, fill_value=0)
    return train_aligned.astype(np.float32), test_aligned.astype(np.float32)


def stratified_split(
    frame: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target],
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    pred_arr = np.asarray(predictions)
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, pred_arr, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true_arr, pred_arr, zero_division=0, average="binary" if np.unique(y_true_arr).size <= 2 else "macro")),
        "recall": float(recall_score(y_true_arr, pred_arr, zero_division=0, average="binary" if np.unique(y_true_arr).size <= 2 else "macro")),
    }
    if probabilities is not None and np.unique(y_true_arr).size == 2:
        prob_arr = np.asarray(probabilities, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, prob_arr))
        except Exception:
            pass
        try:
            metrics["average_precision"] = float(average_precision_score(y_true_arr, prob_arr))
        except Exception:
            pass
        metrics["brier"] = float(np.mean((prob_arr - y_true_arr) ** 2))
    return metrics


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(predictions, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, pred_arr)),
        "r2": float(r2_score(y_true_arr, pred_arr)),
    }