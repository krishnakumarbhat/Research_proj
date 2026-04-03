from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable
import random

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA_CACHE = ROOT / "data_cache"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
DEVICE = torch.device("cpu")


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


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def maybe_downsample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int | None,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows is None or len(x) <= max_rows:
        return x.copy(), y.copy()

    rng = np.random.default_rng(random_state)
    if y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
        indices: list[int] = []
        classes, counts = np.unique(y, return_counts=True)
        for label, count in zip(classes, counts, strict=True):
            label_indices = np.where(y == label)[0]
            take = max(1, int(round(max_rows * count / len(y))))
            take = min(take, len(label_indices))
            indices.extend(rng.choice(label_indices, size=take, replace=False).tolist())
        if len(indices) > max_rows:
            indices = rng.choice(np.array(indices), size=max_rows, replace=False).tolist()
        indices = np.array(sorted(indices))
        return x[indices].copy(), y[indices].copy()

    indices = rng.choice(len(x), size=max_rows, replace=False)
    return x[indices].copy(), y[indices].copy()


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            pass
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def anomaly_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    if threshold is None:
        threshold = float(np.quantile(scores, 0.9))
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        pass
    return metrics


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
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.ndim == 1 and np.issubdtype(y.dtype, np.integer) else None,
    )
    return x_train, x_test, y_train, y_test


def make_loader(
    x: np.ndarray,
    y: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if y is None:
        dataset = TensorDataset(x_tensor)
    else:
        y_dtype = torch.long if np.issubdtype(np.asarray(y).dtype, np.integer) else torch.float32
        y_tensor = torch.tensor(y, dtype=y_dtype)
        dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_optimizer(model: nn.Module, name: str, *, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    lowered = name.lower()
    if lowered == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lowered == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def apply_weight_quantization(model: nn.Module, bits: int) -> None:
    if bits >= 16:
        return
    qmax = float((2 ** (bits - 1)) - 1)
    with torch.no_grad():
        for parameter in model.parameters():
            scale = parameter.abs().max().item()
            if scale == 0:
                continue
            quantized = torch.round(parameter / scale * qmax) / qmax * scale
            parameter.copy_(quantized)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimated_model_size_kb(model: nn.Module, bits: int = 32) -> float:
    return parameter_count(model) * max(bits, 1) / 8.0 / 1024.0


def inference_latency_ms(model: nn.Module, sample: np.ndarray, runs: int = 20) -> float:
    model.eval()
    sample_tensor = torch.tensor(sample[: min(len(sample), 32)], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        start = perf_counter()
        for _ in range(runs):
            _ = model(sample_tensor)
        elapsed = perf_counter() - start
    return elapsed * 1000.0 / runs


def fit_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    quantize_bits: int | None = None,
) -> tuple[dict[str, float], float]:
    set_seed()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, optimizer_name, lr=lr, weight_decay=weight_decay)
    loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)

    start = perf_counter()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        if quantize_bits is not None:
            apply_weight_quantization(model, quantize_bits)
    fit_seconds = perf_counter() - start

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    score = probs[:, 1] if probs.shape[1] == 2 else None
    metrics = classification_metrics(y_test, y_pred, score)
    metrics["params"] = float(parameter_count(model))
    metrics["model_kb"] = estimated_model_size_kb(model, bits=quantize_bits or 32)
    metrics["latency_ms"] = inference_latency_ms(model, x_test)
    return metrics, fit_seconds


def fit_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
) -> tuple[dict[str, float], float]:
    set_seed()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = make_optimizer(model, optimizer_name, lr=lr)
    loader = make_loader(x_train, y_train.astype(np.float32), batch_size=batch_size, shuffle=True)

    start = perf_counter()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).squeeze(-1)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
    fit_seconds = perf_counter() - start

    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(x_test, dtype=torch.float32, device=DEVICE)).squeeze(-1).cpu().numpy()
    metrics = regression_metrics(y_test, prediction)
    metrics["params"] = float(parameter_count(model))
    metrics["latency_ms"] = inference_latency_ms(model, x_test)
    return metrics, fit_seconds


def fit_autoencoder(
    model: nn.Module,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    healthy_mask: np.ndarray | None = None,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple[dict[str, float], float]:
    set_seed()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_x = x_train if healthy_mask is None else x_train[healthy_mask]
    loader = make_loader(train_x, None, batch_size=batch_size, shuffle=True)

    start = perf_counter()
    model.train()
    for _ in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
    fit_seconds = perf_counter() - start

    model.eval()
    with torch.no_grad():
        train_tensor = torch.tensor(train_x, dtype=torch.float32, device=DEVICE)
        train_scores = ((model(train_tensor) - train_tensor) ** 2).reshape(len(train_x), -1).mean(dim=1).cpu().numpy()
        test_tensor = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
        test_scores = ((model(test_tensor) - test_tensor) ** 2).reshape(len(x_test), -1).mean(dim=1).cpu().numpy()
    threshold = float(np.quantile(train_scores, 0.95))
    metrics = anomaly_metrics(y_test, test_scores, threshold)
    metrics["params"] = float(parameter_count(model))
    metrics["model_kb"] = estimated_model_size_kb(model)
    return metrics, fit_seconds


def fit_distilled_classifier(
    teacher: nn.Module,
    student: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 2.0,
    alpha: float = 0.7,
) -> tuple[dict[str, float], float]:
    set_seed()
    teacher.to(DEVICE)
    student.to(DEVICE)
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(torch.tensor(x_train, dtype=torch.float32, device=DEVICE)).cpu()

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        teacher_logits,
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    start = perf_counter()
    student.train()
    for _ in range(epochs):
        for batch_x, batch_y, batch_teacher_logits in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_teacher_logits = batch_teacher_logits.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(batch_x)
            soft_target = torch.softmax(batch_teacher_logits / temperature, dim=1)
            distillation = kl_loss(
                torch.log_softmax(student_logits / temperature, dim=1),
                soft_target,
            ) * (temperature ** 2)
            hard = ce_loss(student_logits, batch_y)
            loss = alpha * distillation + (1.0 - alpha) * hard
            loss.backward()
            optimizer.step()
    fit_seconds = perf_counter() - start

    student.eval()
    with torch.no_grad():
        logits = student(torch.tensor(x_test, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    score = probs[:, 1] if probs.shape[1] == 2 else None
    metrics = classification_metrics(y_test, y_pred, score)
    metrics["params"] = float(parameter_count(student))
    metrics["model_kb"] = estimated_model_size_kb(student)
    return metrics, fit_seconds


def fgsm_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, epsilon: float = 0.05) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.long, device=DEVICE)
    logits = model(x_tensor)
    loss = criterion(logits, y_tensor)
    loss.backward()
    adversarial = torch.clamp(x_tensor + epsilon * x_tensor.grad.sign(), 0.0, 1.0)
    with torch.no_grad():
        prediction = model(adversarial).argmax(dim=1).cpu().numpy()
    return float(accuracy_score(y, prediction))
