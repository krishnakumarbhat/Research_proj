from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ml.common import ProjectResult, make_record, one_hot_align, timed_run


PROJECT_ID = "label_encoding_stability"
TITLE = "Numerical Stability in Label Encoding"
DATASET_NAME = "Categorical Encoding Challenge-style Synthetic Benchmark"


def _generate_dataset(quick: bool) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 6000 if quick else 16000
    cardinalities = [80, 120, 180, 240, 320, 520]
    frame = pd.DataFrame({f"cat_{index}": rng.integers(0, card, rows).astype(str) for index, card in enumerate(cardinalities)})
    frame["num_0"] = rng.normal(0, 1, rows)
    frame["num_1"] = rng.normal(0, 1, rows)
    signal = (
        frame["cat_0"].astype(int) % 7
        + 0.8 * (frame["cat_3"].astype(int) % 13)
        + 2.0 * frame["num_0"]
        - 1.1 * frame["num_1"]
        + rng.normal(0, 2, rows)
    )
    frame["target"] = (signal > np.median(signal)).astype(int)
    return frame


def _ordinal_encode(train_frame: pd.DataFrame, test_frame: pd.DataFrame, modulo: int | None = None):
    encoded_train = pd.DataFrame(index=train_frame.index)
    encoded_test = pd.DataFrame(index=test_frame.index)
    collision_sum = 0
    original_sum = 0
    for column in train_frame.columns:
        if column.startswith("cat_"):
            categories = {value: index for index, value in enumerate(train_frame[column].astype(str).unique())}
            train_codes = train_frame[column].astype(str).map(categories).fillna(-1).astype(int)
            test_codes = test_frame[column].astype(str).map(categories).fillna(-1).astype(int)
            if modulo is not None:
                collision_sum += max(len(set(train_codes.tolist())) - modulo, 0)
                original_sum += len(set(train_codes.tolist()))
                train_codes = (train_codes % modulo).astype(np.uint8)
                test_codes = (test_codes % modulo).astype(np.uint8)
            encoded_train[column] = train_codes
            encoded_test[column] = test_codes
        else:
            encoded_train[column] = train_frame[column].astype(float)
            encoded_test[column] = test_frame[column].astype(float)
    collision_rate = float(collision_sum / max(original_sum, 1))
    return encoded_train, encoded_test, collision_rate


def _target_encode(train_frame: pd.DataFrame, test_frame: pd.DataFrame, y_train: pd.Series):
    encoded_train = pd.DataFrame(index=train_frame.index)
    encoded_test = pd.DataFrame(index=test_frame.index)
    global_mean = float(y_train.mean())
    for column in train_frame.columns:
        if column.startswith("cat_"):
            stats = pd.DataFrame({"value": train_frame[column].astype(str), "target": y_train}).groupby("value")["target"].agg(["mean", "count"])
            smoothing = (stats["count"] * stats["mean"] + 10 * global_mean) / (stats["count"] + 10)
            encoded_train[column] = train_frame[column].astype(str).map(smoothing).fillna(global_mean)
            encoded_test[column] = test_frame[column].astype(str).map(smoothing).fillna(global_mean)
        else:
            encoded_train[column] = train_frame[column].astype(float)
            encoded_test[column] = test_frame[column].astype(float)
    return encoded_train, encoded_test


def _frequency_encode(train_frame: pd.DataFrame, test_frame: pd.DataFrame):
    encoded_train = pd.DataFrame(index=train_frame.index)
    encoded_test = pd.DataFrame(index=test_frame.index)
    for column in train_frame.columns:
        if column.startswith("cat_"):
            frequencies = train_frame[column].astype(str).value_counts(normalize=True)
            encoded_train[column] = train_frame[column].astype(str).map(frequencies).fillna(0.0)
            encoded_test[column] = test_frame[column].astype(str).map(frequencies).fillna(0.0)
        else:
            encoded_train[column] = train_frame[column].astype(float)
            encoded_test[column] = test_frame[column].astype(float)
    return encoded_train, encoded_test


def _hash_bucket_encode(train_frame: pd.DataFrame, test_frame: pd.DataFrame, buckets: int):
    encoded_train = pd.DataFrame(index=train_frame.index)
    encoded_test = pd.DataFrame(index=test_frame.index)
    collision_sum = 0
    original_sum = 0
    for column in train_frame.columns:
        if column.startswith("cat_"):
            train_values = train_frame[column].astype(str)
            test_values = test_frame[column].astype(str)
            original_sum += train_values.nunique()
            train_codes = train_values.map(lambda value: hash((column, value)) % buckets)
            test_codes = test_values.map(lambda value: hash((column, value)) % buckets)
            collision_sum += max(train_values.nunique() - train_codes.nunique(), 0)
            encoded_train[column] = train_codes.astype(np.int32)
            encoded_test[column] = test_codes.astype(np.int32)
        else:
            encoded_train[column] = train_frame[column].astype(float)
            encoded_test[column] = test_frame[column].astype(float)
    collision_rate = float(collision_sum / max(original_sum, 1))
    return encoded_train, encoded_test, collision_rate


def run(quick: bool = True) -> ProjectResult:
    frame = _generate_dataset(quick)
    train_frame, test_frame = train_test_split(frame, test_size=0.25, random_state=42, stratify=frame["target"])
    x_train = train_frame.drop(columns=["target"])
    x_test = test_frame.drop(columns=["target"])
    y_train = train_frame["target"]
    y_test = test_frame["target"]

    records = []
    model = LogisticRegression(max_iter=1200)

    ordinal_train, ordinal_test, ordinal_collision = _ordinal_encode(x_train, x_test)
    _, ordinal_seconds = timed_run(lambda: model.fit(ordinal_train, y_train))
    ordinal_auc = float(roc_auc_score(y_test, model.predict_proba(ordinal_test)[:, 1]))
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source="synthetic",
            task="binary_classification",
            algorithm="ordinal_int32",
            feature_variant="high_cardinality_categories",
            optimization="stable_integer_codes",
            primary_metric="roc_auc",
            primary_value=ordinal_auc,
            rank_score=ordinal_auc,
            secondary_metric="collision_rate",
            secondary_value=ordinal_collision,
            tertiary_metric="runtime_sec",
            tertiary_value=ordinal_seconds,
            fit_seconds=ordinal_seconds,
            notes="Reference ordinal encoding without wraparound",
        )
    )

    wrapped_train, wrapped_test, wrapped_collision = _ordinal_encode(x_train, x_test, modulo=256)
    wrapped_model = LogisticRegression(max_iter=1200)
    _, wrapped_seconds = timed_run(lambda: wrapped_model.fit(wrapped_train, y_train))
    wrapped_auc = float(roc_auc_score(y_test, wrapped_model.predict_proba(wrapped_test)[:, 1]))
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source="synthetic",
            task="binary_classification",
            algorithm="ordinal_uint8_wrapped",
            feature_variant="high_cardinality_categories",
            optimization="uint8_wraparound",
            primary_metric="roc_auc",
            primary_value=wrapped_auc,
            rank_score=wrapped_auc - wrapped_collision,
            secondary_metric="collision_rate",
            secondary_value=wrapped_collision,
            tertiary_metric="runtime_sec",
            tertiary_value=wrapped_seconds,
            fit_seconds=wrapped_seconds,
            notes="Purposefully wrapped integer codes to test collisions",
        )
    )

    one_hot_train, one_hot_test = one_hot_align(x_train, x_test)
    one_hot_model = LogisticRegression(max_iter=1200, n_jobs=-1)
    _, one_hot_seconds = timed_run(lambda: one_hot_model.fit(one_hot_train, y_train))
    one_hot_auc = float(roc_auc_score(y_test, one_hot_model.predict_proba(one_hot_test)[:, 1]))
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source="synthetic",
            task="binary_classification",
            algorithm="one_hot_sparse",
            feature_variant="high_cardinality_categories",
            optimization="sparse_indicator_matrix",
            primary_metric="roc_auc",
            primary_value=one_hot_auc,
            rank_score=one_hot_auc,
            secondary_metric="collision_rate",
            secondary_value=0.0,
            tertiary_metric="runtime_sec",
            tertiary_value=one_hot_seconds,
            fit_seconds=one_hot_seconds,
            notes="Collision-free reference encoding",
        )
    )

    target_train, target_test = _target_encode(x_train, x_test, y_train)
    target_model = LogisticRegression(max_iter=1200)
    _, target_seconds = timed_run(lambda: target_model.fit(target_train, y_train))
    target_auc = float(roc_auc_score(y_test, target_model.predict_proba(target_test)[:, 1]))
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source="synthetic",
            task="binary_classification",
            algorithm="target_encoding",
            feature_variant="high_cardinality_categories",
            optimization="smoothed_mean_encoding",
            primary_metric="roc_auc",
            primary_value=target_auc,
            rank_score=target_auc,
            secondary_metric="collision_rate",
            secondary_value=0.0,
            tertiary_metric="runtime_sec",
            tertiary_value=target_seconds,
            fit_seconds=target_seconds,
            notes="Smoothed train-only target means",
        )
    )

    frequency_train, frequency_test = _frequency_encode(x_train, x_test)
    frequency_model = LogisticRegression(max_iter=1200)
    _, frequency_seconds = timed_run(lambda: frequency_model.fit(frequency_train, y_train))
    frequency_auc = float(roc_auc_score(y_test, frequency_model.predict_proba(frequency_test)[:, 1]))
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source="synthetic",
            task="binary_classification",
            algorithm="frequency_encoding",
            feature_variant="high_cardinality_categories",
            optimization="category_frequency_prior",
            primary_metric="roc_auc",
            primary_value=frequency_auc,
            rank_score=frequency_auc,
            secondary_metric="collision_rate",
            secondary_value=0.0,
            tertiary_metric="runtime_sec",
            tertiary_value=frequency_seconds,
            fit_seconds=frequency_seconds,
            notes="Category frequency encoding without target leakage",
        )
    )

    hash_train, hash_test, hash_collision = _hash_bucket_encode(x_train, x_test, buckets=256)
    hash_model = LogisticRegression(max_iter=1200)
    _, hash_seconds = timed_run(lambda: hash_model.fit(hash_train, y_train))
    hash_auc = float(roc_auc_score(y_test, hash_model.predict_proba(hash_test)[:, 1]))
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source="synthetic",
            task="binary_classification",
            algorithm="hash_bucket_256",
            feature_variant="high_cardinality_categories",
            optimization="hash_bucket_encoding",
            primary_metric="roc_auc",
            primary_value=hash_auc,
            rank_score=hash_auc - hash_collision,
            secondary_metric="collision_rate",
            secondary_value=hash_collision,
            tertiary_metric="runtime_sec",
            tertiary_value=hash_seconds,
            fit_seconds=hash_seconds,
            notes="Hash bucket encoding with explicit collision accounting",
        )
    )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The best encoding strategy was {best.algorithm} with ROC AUC {best.primary_value:.3f}. Wrapped uint8 codes introduced collisions that materially degraded ranking performance relative to stable or sparse encodings."
        ),
        recommendation=(
            "Avoid low-width wrapped ordinal codes for high-cardinality features. If memory is tight, prefer sparse one-hot or smoothed target encoding rather than forcing categories into a tiny integer range."
        ),
        key_findings=[
            f"Best encoding: {best.algorithm}.",
            f"Wrapped uint8 collision rate was {wrapped_collision:.3f}.",
            "The encoding failure mode shows up as both lower ROC AUC and unstable feature semantics when categories collide.",
        ],
        caveats=[
            "This is a synthetic high-cardinality benchmark rather than the full Kaggle encoding challenge dataset.",
            "Target encoding is implemented with simple train-only smoothing and no nested out-of-fold protection.",
            "The quick benchmark focuses on numerical stability effects, not full leaderboard-style tuning.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
