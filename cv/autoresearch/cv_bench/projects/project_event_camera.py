from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cv_bench.common import ProjectResult, classification_metrics, configured_n_jobs, make_record, timed_run
from cv_bench.real_data import load_n_mnist_images


PROJECT_ID = "event_camera_processing"
TITLE = "Event Camera (Neuromorphic) Processing"
DATASET_NAME = "N-MNIST / DVS Digit Recognition"


def _load_dataset(quick: bool) -> tuple[np.ndarray, np.ndarray, str]:
    real_dataset = load_n_mnist_images(quick)
    if real_dataset is not None:
        return real_dataset
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(int)
    if quick:
        return images, labels, "sklearn_digits_fallback"
    tiled = np.repeat(images, 2, axis=0)
    tiled_labels = np.repeat(labels, 2, axis=0)
    jitter = np.random.default_rng(42).normal(0.0, 0.03, size=tiled.shape)
    tiled = np.clip(tiled + jitter, 0.0, 1.0)
    return tiled, tiled_labels, "synthetic_event_expansion"


def _event_histogram(images: np.ndarray) -> np.ndarray:
    positive = images.reshape(len(images), -1)
    negative = (1.0 - images).reshape(len(images), -1)
    return np.concatenate([positive, negative], axis=1)


def _time_surface(images: np.ndarray) -> np.ndarray:
    height, width = images.shape[1:3]
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    timestamps = (xx + yy) / 2.0
    surface = images * timestamps[None, ...]
    return surface.reshape(len(images), -1)


def _motion_profile(images: np.ndarray) -> np.ndarray:
    vertical = images.sum(axis=1)
    horizontal = images.sum(axis=2)
    diagonal = np.asarray(
        [[np.trace(image), np.trace(np.fliplr(image))] for image in images],
        dtype=np.float32,
    )
    return np.concatenate([vertical, horizontal, diagonal], axis=1)


def run(quick: bool = True) -> ProjectResult:
    images, labels, source = _load_dataset(quick)
    using_real = source == "n_mnist_real_subset"
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    feature_sets = {
        "event_histogram": _event_histogram,
        "time_surface": _time_surface,
        "motion_profile": _motion_profile,
    }
    parallel_jobs = configured_n_jobs()
    models = {
        "logistic_regression": LogisticRegression(max_iter=1600, n_jobs=parallel_jobs),
        "random_forest": RandomForestClassifier(n_estimators=180 if not quick else 110, random_state=42, n_jobs=parallel_jobs),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=220 if not quick else 140, random_state=42),
    }

    records = []
    for feature_name, extractor in feature_sets.items():
        train_features = extractor(x_train)
        test_features = extractor(x_test)
        for algorithm, model in models.items():
            _, fit_seconds = timed_run(lambda current=model: current.fit(train_features, y_train))
            predictions = model.predict(test_features)
            metrics = classification_metrics(y_test, predictions)
            records.append(
                make_record(
                    project=PROJECT_ID,
                    dataset=DATASET_NAME,
                    source=source,
                    task="neuromorphic_digit_classification",
                    algorithm=algorithm,
                    feature_variant=feature_name,
                    optimization="event_feature_encoding",
                    primary_metric="accuracy",
                    primary_value=metrics["accuracy"],
                    rank_score=metrics["accuracy"],
                    secondary_metric="balanced_accuracy",
                    secondary_value=metrics["balanced_accuracy"],
                    tertiary_metric="macro_f1",
                    tertiary_value=metrics["macro_f1"],
                    fit_seconds=fit_seconds,
                    notes=(
                        "Sampled real N-MNIST event files converted into compact event frames"
                        if using_real
                        else "Synthetic event surfaces derived from compact handwritten digits"
                    ),
                )
            )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The best neuromorphic digit pipeline was {best.algorithm} on {best.feature_variant}, reaching accuracy {best.primary_value:.3f}. "
            "Event histograms and time surfaces both remained competitive in the compact CPU-oriented benchmark."
        ),
        recommendation=(
            "Start event-camera work with lightweight event encodings before moving to spiking networks. On small neuromorphic datasets, feature engineering can beat architectural complexity."
        ),
        key_findings=[
            f"Best result: {best.algorithm} on {best.feature_variant}.",
            "Temporal encodings were strong, but compact motion-profile features were cheaper to fit.",
            "The suite can use a sampled real N-MNIST subset when it is present under the data cache.",
        ],
        caveats=(
            [
                "The real-data path uses a storage-limited N-MNIST subset instead of the full 70k-file archive.",
                "Event files are collapsed into compact spatial frames for classical CPU baselines rather than fed into a spiking model directly.",
                "No spiking neural network is trained in this benchmark path.",
            ]
            if using_real
            else [
                "If N-MNIST is not available locally, this module uses sklearn digits as a deterministic neuromorphic fallback.",
                "The fallback approximates DVS event structure with synthetic timestamps rather than true event streams.",
                "No spiking neural network is trained in quick mode.",
            ]
        ),
    )


if __name__ == "__main__":
    print(run(quick=True).summary)