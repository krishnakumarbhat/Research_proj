from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ml.common import ProjectResult, make_record, one_hot_align, timed_run


PROJECT_ID = "tabular_meta_learning"
TITLE = "Tabular Meta-Learning"
DATASET_NAME = "OpenML-style Small Classification Suite"


def _load_suite() -> list[tuple[str, pd.DataFrame, str, str]]:
    suite: list[tuple[str, pd.DataFrame, str, str]] = []
    builtins = [
        ("iris", load_iris(as_frame=True), "sklearn"),
        ("wine", load_wine(as_frame=True), "sklearn"),
        ("breast_cancer", load_breast_cancer(as_frame=True), "sklearn"),
    ]
    for name, bundle, source in builtins:
        frame = bundle.frame.copy()
        target = bundle.target.name
        suite.append((name, frame, target, source))

    openml_candidates = [
        ("blood-transfusion-service-center", 1),
        ("phoneme", 1),
        ("diabetes", 1),
    ]
    for name, version in openml_candidates:
        try:
            dataset = fetch_openml(name=name, version=version, as_frame=True)
            frame = dataset.frame.copy()
            target_name = dataset.target.name
            frame[target_name] = pd.Series(dataset.target).astype(str)
            suite.append((name, frame, target_name, "openml"))
        except Exception:
            continue
    return suite


def _meta_features(frame: pd.DataFrame, target: str) -> np.ndarray:
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include=[np.number]).shape[1]
    categorical = features.shape[1] - numeric
    target_counts = frame[target].astype(str).value_counts(normalize=True)
    entropy = float(-(target_counts * np.log2(target_counts + 1e-12)).sum())
    missing_ratio = float(features.isna().mean().mean())
    return np.asarray(
        [
            len(frame),
            features.shape[1],
            numeric,
            categorical,
            float(features.shape[1] / max(len(frame), 1)),
            entropy,
            missing_ratio,
        ],
        dtype=float,
    )


def _prepare_xy(frame: pd.DataFrame, target: str):
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=42,
        stratify=frame[target].astype(str),
    )
    x_train, x_test = one_hot_align(train_frame.drop(columns=[target]), test_frame.drop(columns=[target]))
    y_train = train_frame[target].astype(str)
    y_test = test_frame[target].astype(str)
    return x_train, x_test, y_train, y_test


def run(quick: bool = True) -> ProjectResult:
    suite = _load_suite()
    if quick and len(suite) > 5:
        suite = suite[:5]

    models = {
        "logistic_regression": LogisticRegression(max_iter=1200, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=180 if not quick else 100, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=220 if not quick else 140, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    dataset_scores: dict[str, dict[str, float]] = {}
    dataset_meta: dict[str, np.ndarray] = {}
    records = []
    for dataset_name, frame, target, source in suite:
        dataset_meta[dataset_name] = _meta_features(frame, target)
        x_train, x_test, y_train, y_test = _prepare_xy(frame, target)
        dataset_scores[dataset_name] = {}
        for algorithm, model in models.items():
            _, fit_seconds = timed_run(lambda current=model: current.fit(x_train, y_train))
            predictions = model.predict(x_test)
            score = float(balanced_accuracy_score(y_test, predictions))
            dataset_scores[dataset_name][algorithm] = score
            records.append(
                make_record(
                    project=PROJECT_ID,
                    dataset=DATASET_NAME,
                    source=source,
                    task="meta_learning_benchmark",
                    algorithm=algorithm,
                    feature_variant=dataset_name,
                    optimization="holdout_balanced_accuracy",
                    primary_metric="balanced_accuracy",
                    primary_value=score,
                    rank_score=score,
                    secondary_metric="dataset_size",
                    secondary_value=float(len(frame)),
                    tertiary_metric="feature_count",
                    tertiary_value=float(frame.drop(columns=[target]).shape[1]),
                    fit_seconds=fit_seconds,
                    notes=f"Per-dataset score on {dataset_name}",
                )
            )

    meta_hits = 0
    regrets = []
    for held_out in dataset_meta:
        held_vector = dataset_meta[held_out]
        best_neighbor = None
        best_distance = float("inf")
        for other_name, other_vector in dataset_meta.items():
            if other_name == held_out:
                continue
            distance = float(np.linalg.norm((held_vector - other_vector) / (np.std(np.vstack(list(dataset_meta.values())), axis=0) + 1e-6)))
            if distance < best_distance:
                best_distance = distance
                best_neighbor = other_name
        assert best_neighbor is not None
        recommended = max(dataset_scores[best_neighbor], key=dataset_scores[best_neighbor].get)
        actual_best = max(dataset_scores[held_out], key=dataset_scores[held_out].get)
        meta_hits += int(recommended == actual_best)
        regret = dataset_scores[held_out][actual_best] - dataset_scores[held_out][recommended]
        regrets.append(regret)

    recommendation_accuracy = meta_hits / max(len(dataset_meta), 1)
    mean_regret = float(np.mean(regrets)) if regrets else 0.0
    average_algorithm_scores = {
        algorithm: float(np.mean([scores[algorithm] for scores in dataset_scores.values()]))
        for algorithm in models
    }
    best_average_algorithm = max(average_algorithm_scores, key=average_algorithm_scores.get)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source="meta_layer",
            task="meta_learning_benchmark",
            algorithm="nearest_dataset_meta_selector",
            feature_variant="leave_one_dataset_out",
            optimization="meta_feature_neighbor_match",
            primary_metric="recommendation_accuracy",
            primary_value=recommendation_accuracy,
            rank_score=recommendation_accuracy - mean_regret,
            secondary_metric="mean_regret",
            secondary_value=mean_regret,
            tertiary_metric="dataset_count",
            tertiary_value=float(len(dataset_meta)),
            fit_seconds=0.0,
            notes=f"Best average base learner: {best_average_algorithm}",
        )
    )

    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"Across {len(dataset_meta)} small classification tasks, the best average base learner was {best_average_algorithm}. "
            f"A nearest-dataset meta-selector recovered the true best learner on {recommendation_accuracy:.3f} of held-out datasets with mean regret {mean_regret:.3f}."
        ),
        recommendation=(
            "If you are benchmarking tiny tabular datasets, store meta-features and prior model rankings. Even a simple nearest-dataset recommender can cut the search space before full hyperparameter tuning."
        ),
        key_findings=[
            f"The strongest average learner across the suite was {best_average_algorithm}.",
            f"Meta-selection accuracy reached {recommendation_accuracy:.3f} on leave-one-dataset-out evaluation.",
            "Meta-features like sample count, feature count, missingness, and label entropy are enough to build a lightweight warm-start policy.",
        ],
        caveats=[
            "This runner uses a compact OpenML-style suite and built-in datasets rather than the full CC18 benchmark.",
            "Scores come from single train/test splits per dataset, not repeated folds.",
            "The meta-selector is deliberately simple so it stays CPU-only and transparent.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
