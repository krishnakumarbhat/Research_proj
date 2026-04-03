from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_CACHE = ROOT / "data_cache"
RESULTS_DIR = ROOT / "results"
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
    secondary_metric: str = ""
    secondary_value: float | None = None
    tertiary_metric: str = ""
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
    records: list[ExperimentRecord]
    summary: str
    recommendation: str
    key_findings: list[str]
    caveats: list[str]


def ensure_directories() -> None:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timed_run(fn: Callable[[], object]) -> tuple[object, float]:
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def download_to_cache(url: str, relative_path: str, timeout: int = 60) -> Path | None:
    ensure_directories()
    destination = DATA_CACHE / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None

    destination.write_bytes(response.content)
    return destination


def existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def maybe_downsample(
    frame: pd.DataFrame,
    *,
    max_rows: int | None,
    stratify: str | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy()

    if stratify and stratify in frame.columns:
        sampled_parts = []
        for _, group in frame.groupby(stratify):
            group_sample_size = min(len(group), max(1, int(max_rows * len(group) / len(frame))))
            sampled_parts.append(group.sample(group_sample_size, random_state=random_state))
        sampled = pd.concat(sampled_parts, axis=0)
        return sampled.sample(min(max_rows, len(sampled)), random_state=random_state).reset_index(drop=True)

    return frame.sample(max_rows, random_state=random_state).reset_index(drop=True)


def one_hot_align(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train_features.assign(_split_marker="train"), test_features.assign(_split_marker="test")],
        axis=0,
    )
    encoded = pd.get_dummies(combined, dummy_na=True)
    train_encoded = encoded.loc[encoded["_split_marker_train"] == 1].drop(
        columns=["_split_marker_train", "_split_marker_test"],
        errors="ignore",
    )
    test_encoded = encoded.loc[encoded["_split_marker_test"] == 1].drop(
        columns=["_split_marker_train", "_split_marker_test"],
        errors="ignore",
    )
    train_encoded.index = train_features.index
    test_encoded.index = test_features.index
    return train_encoded, test_encoded


def build_preprocessor(
    frame: pd.DataFrame,
    target: str,
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    features = frame.drop(columns=[target])
    numeric_columns = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass

        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            pass

        try:
            metrics["brier"] = float(brier_score_loss(y_true, y_score))
        except ValueError:
            pass

    return metrics


def interval_metrics(
    y_true: pd.Series | np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    inside = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(np.mean(inside)),
        "avg_width": float(np.mean(upper - lower)),
    }


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
    fit_seconds: float,
    notes: str = "",
    secondary_metric: str = "",
    secondary_value: float | None = None,
    tertiary_metric: str = "",
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


def stratified_split(
    frame: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_index, test_index = train_test_split(
        frame.index,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target],
    )
    return frame.loc[train_index].reset_index(drop=True), frame.loc[test_index].reset_index(drop=True)
