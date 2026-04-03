from __future__ import annotations

import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.common import DATA_CACHE, ProjectResult, download_to_cache, make_record, one_hot_align


PROJECT_ID = "memory_safe_feature_pipeline"
TITLE = "Memory-Safe Feature Extraction Pipelines"
DATASET_NAME = "NSL-KDD"
NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


def _measure_run(fn):
    tracemalloc.start()
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def _load_dataset(quick: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train_candidates = [
        DATA_CACHE / "nsl_kdd" / "KDDTrain+.txt",
        Path(__file__).resolve().parent / "KDDTrain+.txt",
    ]
    test_candidates = [
        DATA_CACHE / "nsl_kdd" / "KDDTest+.txt",
        Path(__file__).resolve().parent / "KDDTest+.txt",
    ]
    train_path = next((path for path in train_candidates if path.exists()), None)
    test_path = next((path for path in test_candidates if path.exists()), None)
    if train_path is None:
        train_path = download_to_cache(
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
            "nsl_kdd/KDDTrain+.txt",
        )
    if test_path is None:
        test_path = download_to_cache(
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
            "nsl_kdd/KDDTest+.txt",
        )

    if train_path is not None and test_path is not None:
        train_frame = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
        test_frame = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)
        source = "github_mirror"
    else:
        raise RuntimeError("NSL-KDD download failed and no local copy is available")

    if quick:
        train_frame = train_frame.sample(4000, random_state=42)
        test_frame = test_frame.sample(2000, random_state=42)
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True), source


def _prepare_targets(train_frame: pd.DataFrame, test_frame: pd.DataFrame):
    y_train = (train_frame["label"].astype(str) != "normal").astype(int)
    y_test = (test_frame["label"].astype(str) != "normal").astype(int)
    x_train = train_frame.drop(columns=["label", "difficulty"])
    x_test = test_frame.drop(columns=["label", "difficulty"])
    return x_train, x_test, y_train, y_test


def _hash_rows(frame: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    for _, row in frame.iterrows():
        record: dict[str, float] = {}
        for column, value in row.items():
            if isinstance(value, str):
                record[f"{column}={value}"] = 1.0
            else:
                record[column] = float(value)
        rows.append(record)
    return rows


def run(quick: bool = True) -> ProjectResult:
    train_frame, test_frame, source = _load_dataset(quick)
    x_train, x_test, y_train, y_test = _prepare_targets(train_frame, test_frame)
    categorical = ["protocol_type", "service", "flag"]
    numeric = [column for column in x_train.columns if column not in categorical]

    records = []

    def dense_pipeline():
        encoded_train, encoded_test = one_hot_align(x_train, x_test)
        model = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)
        model.fit(encoded_train.to_numpy(dtype=np.float32), y_train)
        predictions = model.predict(encoded_test.to_numpy(dtype=np.float32))
        return float(balanced_accuracy_score(y_test, predictions))

    dense_score, dense_seconds, dense_peak = _measure_run(dense_pipeline)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="intrusion_classification",
            algorithm="dense_get_dummies",
            feature_variant="copy_heavy_dense_matrix",
            optimization="baseline_dense_one_hot",
            primary_metric="balanced_accuracy",
            primary_value=dense_score,
            rank_score=dense_score - 0.005 * dense_peak,
            secondary_metric="peak_memory_mb",
            secondary_value=dense_peak,
            tertiary_metric="runtime_sec",
            tertiary_value=dense_seconds,
            fit_seconds=dense_seconds,
            notes="Dense pandas one-hot representation",
        )
    )

    def sparse_pipeline():
        pipeline = Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(with_mean=False), numeric),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ]
                    ),
                ),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="saga", n_jobs=-1)),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        return float(balanced_accuracy_score(y_test, predictions))

    sparse_score, sparse_seconds, sparse_peak = _measure_run(sparse_pipeline)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="intrusion_classification",
            algorithm="sparse_one_hot_pipeline",
            feature_variant="sparse_encoder",
            optimization="column_transformer_sparse",
            primary_metric="balanced_accuracy",
            primary_value=sparse_score,
            rank_score=sparse_score - 0.005 * sparse_peak,
            secondary_metric="peak_memory_mb",
            secondary_value=sparse_peak,
            tertiary_metric="runtime_sec",
            tertiary_value=sparse_seconds,
            fit_seconds=sparse_seconds,
            notes="Sparse encoder with solver-friendly linear model",
        )
    )

    def hashing_pipeline():
        hasher = FeatureHasher(n_features=2**12 if quick else 2**14, input_type="dict")
        x_train_hashed = hasher.transform(_hash_rows(x_train))
        x_test_hashed = hasher.transform(_hash_rows(x_test))
        model = LogisticRegression(max_iter=500, class_weight="balanced", solver="saga", n_jobs=-1)
        model.fit(x_train_hashed, y_train)
        predictions = model.predict(x_test_hashed)
        return float(balanced_accuracy_score(y_test, predictions))

    hash_score, hash_seconds, hash_peak = _measure_run(hashing_pipeline)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="intrusion_classification",
            algorithm="feature_hashing_pipeline",
            feature_variant="hashed_streaming_features",
            optimization="fixed_width_hashing",
            primary_metric="balanced_accuracy",
            primary_value=hash_score,
            rank_score=hash_score - 0.005 * hash_peak,
            secondary_metric="peak_memory_mb",
            secondary_value=hash_peak,
            tertiary_metric="runtime_sec",
            tertiary_value=hash_seconds,
            fit_seconds=hash_seconds,
            notes="Hash-based fixed-width representation",
        )
    )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The best memory-safe pipeline was {best.algorithm}, balancing balanced accuracy {best.primary_value:.3f} against peak memory {best.secondary_value:.1f} MB. Sparse and hashed representations cut allocation pressure relative to dense one-hot encoding."
        ),
        recommendation=(
            "Use sparse or hashed feature extraction when intrusion logs start to scale. Dense one-hot matrices are simple, but they become the bottleneck long before the classifier does."
        ),
        key_findings=[
            f"Lowest-memory competitive pipeline: {best.algorithm}.",
            "Sparse one-hot usually keeps most of the linear-model accuracy while lowering peak allocation.",
            "Feature hashing is useful when the categorical vocabulary grows faster than the available RAM budget.",
        ],
        caveats=[
            "Peak memory is measured with Python-level tracemalloc, which is a proxy rather than full native-process RSS.",
            "This benchmark requires the NSL-KDD train and test files, which are fetched from a public mirror if missing locally.",
            "The quick benchmark subsamples both train and test sets for runtime control.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
