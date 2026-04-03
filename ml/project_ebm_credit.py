from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.common import (
    DATA_CACHE,
    ProjectResult,
    classification_metrics,
    make_record,
    maybe_downsample,
    one_hot_align,
    stratified_split,
    timed_run,
)


PROJECT_ID = "ebm_credit"
TITLE = "Explainable Boosting Machines in High-Stakes Domains"
DATASET_NAME = "Give Me Some Credit / Credit-G"


def _load_dataset(quick: bool) -> tuple[pd.DataFrame, str, str]:
    local_candidates = [
        DATA_CACHE / "give_me_some_credit" / "cs-training.csv",
        Path(__file__).resolve().parent / "cs-training.csv",
    ]
    local_path = next((path for path in local_candidates if path.exists()), None)
    if local_path is not None:
        frame = pd.read_csv(local_path)
        if "Unnamed: 0" in frame.columns:
            frame = frame.drop(columns=["Unnamed: 0"])
        target = "SeriousDlqin2yrs"
        source = "local_kaggle_copy"
    else:
        try:
            openml = fetch_openml(name="credit-g", version=1, as_frame=True)
            frame = openml.frame
            frame["target"] = (frame[openml.target.name].astype(str).str.lower() == "bad").astype(int)
            frame = frame.drop(columns=[openml.target.name])
            target = "target"
            source = "openml_credit_g"
        except Exception:
            cancer = load_breast_cancer(as_frame=True)
            frame = cancer.frame
            target = cancer.target.name
            source = "sklearn_breast_cancer_fallback"

    if quick:
        frame = maybe_downsample(frame, max_rows=6000, stratify=target)
    return frame.reset_index(drop=True), target, source


def _prepare_ebm_frames(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_train = train_frame.copy()
    prepared_test = test_frame.copy()
    for frame in [prepared_train, prepared_test]:
        for column in frame.columns:
            if frame[column].dtype.kind in "biufc":
                frame[column] = frame[column].fillna(frame[column].median())
            else:
                frame[column] = frame[column].astype(str).fillna("missing")
    return prepared_train, prepared_test


def run(quick: bool = True) -> ProjectResult:
    frame, target, source = _load_dataset(quick)
    train_frame, test_frame = stratified_split(frame, target)
    x_train = train_frame.drop(columns=[target])
    x_test = test_frame.drop(columns=[target])
    y_train = train_frame[target].astype(int)
    y_test = test_frame[target].astype(int)

    encoded_train, encoded_test = one_hot_align(x_train, x_test)
    records = []

    baseline_models = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", n_jobs=-1),
        "logistic_elastic_net": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.7,
            max_iter=2500,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240 if not quick else 150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=260 if not quick else 160,
            random_state=42,
        ),
    }

    for name, model in baseline_models.items():
        _, fit_seconds = timed_run(lambda current=model: current.fit(encoded_train, y_train))
        probabilities = model.predict_proba(encoded_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(y_test, predictions, probabilities)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="binary_classification",
                algorithm=name,
                feature_variant="one_hot_tabular",
                optimization="baseline_hyperparameters",
                primary_metric="roc_auc",
                primary_value=metrics.get("roc_auc", metrics["balanced_accuracy"]),
                rank_score=metrics.get("roc_auc", metrics["balanced_accuracy"]),
                secondary_metric="average_precision",
                secondary_value=metrics.get("average_precision", metrics["f1"]),
                tertiary_metric="brier",
                tertiary_value=metrics.get("brier", 0.0),
                fit_seconds=fit_seconds,
                notes="High-capacity black-box baseline",
            )
        )

    ebm_train, ebm_test = _prepare_ebm_frames(x_train, x_test)
    ebm_variants = {
        "ebm_default": ExplainableBoostingClassifier(random_state=42, interactions=0),
        "ebm_interactions": ExplainableBoostingClassifier(
            random_state=42,
            interactions=5,
            outer_bags=12 if not quick else 8,
        ),
        "ebm_interactions_dense": ExplainableBoostingClassifier(
            random_state=42,
            interactions=8 if quick else 12,
            outer_bags=10 if quick else 14,
            learning_rate=0.03,
        ),
    }

    ebm_notes = []
    for name, model in ebm_variants.items():
        _, fit_seconds = timed_run(lambda current=model: current.fit(ebm_train, y_train))
        probabilities = model.predict_proba(ebm_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(y_test, predictions, probabilities)
        importances = model.term_importances()
        term_pairs = sorted(zip(model.term_names_, importances), key=lambda item: item[1], reverse=True)[:3]
        top_terms = ", ".join(f"{term}={importance:.3f}" for term, importance in term_pairs)
        ebm_notes.append(f"{name}: {top_terms}")
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="binary_classification",
                algorithm=name,
                feature_variant="native_mixed_types",
                optimization="interpretable_boosting",
                primary_metric="roc_auc",
                primary_value=metrics.get("roc_auc", metrics["balanced_accuracy"]),
                rank_score=metrics.get("roc_auc", metrics["balanced_accuracy"]),
                secondary_metric="average_precision",
                secondary_value=metrics.get("average_precision", metrics["f1"]),
                tertiary_metric="brier",
                tertiary_value=metrics.get("brier", 0.0),
                fit_seconds=fit_seconds,
                notes=f"Top terms: {top_terms}",
            )
        )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The best credit-risk model was {best.algorithm} with ROC AUC {best.primary_value:.3f}. "
            f"The EBM variants expose dominant risk factors directly, which makes them attractive when auditability matters as much as raw discrimination."
        ),
        recommendation=(
            "Choose an EBM when you need near-ensemble accuracy with feature-level accountability. If a black-box model only wins marginally, the interpretability trade-off usually favors the EBM in regulated credit workflows."
        ),
        key_findings=[
            f"Best observed ROC AUC was {best.primary_value:.3f} from {best.algorithm}.",
            f"EBM global terms remained readable: {ebm_notes[0] if ebm_notes else 'not available'}.",
            "Logistic regression provides a strong transparency baseline but usually leaves accuracy on the table relative to boosting-based models.",
        ],
        caveats=[
            "If the Kaggle credit dataset is not available locally, the runner uses OpenML Credit-G and then a breast-cancer fallback if remote access fails.",
            "This benchmark uses a single stratified split, not repeated cross-validation.",
            "Interpretability is summarized through global term importance rather than full local explanation artifacts.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
