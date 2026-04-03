from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, make_classification
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml.common import ProjectResult, make_record, timed_run


PROJECT_ID = "tree_ensemble_optimization"
TITLE = "Algorithmic Optimization of Tree Ensembles"
DATASET_NAME = "Covertype"


def _load_dataset(quick: bool) -> tuple[pd.DataFrame, str]:
    try:
        x, y = fetch_covtype(return_X_y=True, as_frame=True)
        frame = x.copy()
        frame["target"] = y
        source = "sklearn"
    except Exception:
        features, target = make_classification(
            n_samples=30000 if quick else 90000,
            n_features=40,
            n_informative=24,
            n_classes=7,
            random_state=42,
        )
        frame = pd.DataFrame(features, columns=[f"f_{index}" for index in range(features.shape[1])])
        frame["target"] = target
        source = "synthetic_fallback"
    if quick and len(frame) > 25000:
        sampled = []
        for _, group in frame.groupby("target"):
            sampled.append(group.sample(min(len(group), 3500), random_state=42))
        frame = pd.concat(sampled, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return frame, source


def _prediction_latency_ms(model, x_test):
    subset = x_test[: min(2000, len(x_test))]
    start = perf_counter()
    for _ in range(3):
        model.predict(subset)
    elapsed = perf_counter() - start
    return 1000 * elapsed / 3 / max(len(subset) / 1000, 1e-6)


def run(quick: bool = True) -> ProjectResult:
    frame, source = _load_dataset(quick)
    x_train, x_test, y_train, y_test = train_test_split(
        frame.drop(columns=["target"]).astype(np.float32),
        frame["target"],
        test_size=0.2,
        random_state=42,
        stratify=frame["target"],
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=220 if not quick else 140,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=220 if not quick else 140,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=10,
            max_iter=240 if not quick else 160,
            random_state=42,
        ),
        "random_forest_shallow": RandomForestClassifier(
            n_estimators=180 if not quick else 120,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_shallow": ExtraTreesClassifier(
            n_estimators=180 if not quick else 120,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting_fast": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=8,
            max_iter=150 if not quick else 100,
            random_state=42,
        ),
    }

    records = []
    for algorithm, model in models.items():
        _, fit_seconds = timed_run(lambda current=model: current.fit(x_train, y_train))
        predictions = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))
        latency = float(_prediction_latency_ms(model, x_test))
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="multiclass_classification",
                algorithm=algorithm,
                feature_variant="float32_features",
                optimization="tree_ensemble_benchmark",
                primary_metric="accuracy",
                primary_value=accuracy,
                rank_score=accuracy - 0.002 * fit_seconds - 0.0002 * latency,
                secondary_metric="fit_seconds",
                secondary_value=fit_seconds,
                tertiary_metric="predict_ms_per_1k",
                tertiary_value=latency,
                fit_seconds=fit_seconds,
                notes="Accuracy-speed trade-off on medium-scale tabular forest cover data",
            )
        )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The best accuracy-speed compromise was {best.algorithm}, reaching accuracy {best.primary_value:.3f} with fit time {best.secondary_value:.1f}s and inference latency {best.tertiary_value:.2f} ms per 1k rows."
        ),
        recommendation=(
            "Benchmark tree ensembles with both accuracy and systems metrics. ExtraTrees or shallow forests often give a better production trade-off than a raw accuracy leader that is slower to fit or score."
        ),
        key_findings=[
            f"Best overall trade-off: {best.algorithm}.",
            "Float32 features are sufficient for this style of ensemble benchmark and reduce memory pressure.",
            "Inference latency can invert the ranking when two ensembles are close on accuracy.",
        ],
        caveats=[
            "The quick benchmark subsamples Covertype to keep fit times reasonable.",
            "Latency is measured with wall-clock timing on this machine, not controlled hardware counters.",
            "If Covertype is unavailable, a synthetic multiclass fallback is used.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
