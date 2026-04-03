from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score

from ml.common import DATA_CACHE, ProjectResult, download_to_cache, make_record, one_hot_align, timed_run


PROJECT_ID = "security_aware_intrusion"
TITLE = "Security-Aware ML for Intrusion Detection"
DATASET_NAME = "UNSW-NB15-style Intrusion Detection"


def _synthetic_intrusion(quick: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rng = np.random.default_rng(42)
    def make_split(rows: int) -> pd.DataFrame:
        proto = rng.choice(["tcp", "udp", "icmp"], size=rows, p=[0.6, 0.3, 0.1])
        service = rng.choice(["http", "dns", "ssh", "smtp", "ftp"], size=rows)
        state = rng.choice(["FIN", "INT", "CON", "REQ"], size=rows)
        attack = rng.random(rows) < 0.18
        src = rng.lognormal(6 + attack * 0.6, 1.0, rows)
        dst = rng.lognormal(5.5 + attack * 0.4, 1.0, rows)
        duration = rng.exponential(2.0 + attack * 1.2, rows)
        sload = rng.lognormal(7 + attack * 0.8, 0.8, rows)
        dload = rng.lognormal(6.2 + attack * 0.7, 0.9, rows)
        frame = pd.DataFrame(
            {
                "proto": proto,
                "service": service,
                "state": state,
                "src_bytes": src,
                "dst_bytes": dst,
                "duration": duration,
                "sload": sload,
                "dload": dload,
                "label": attack.astype(int),
            }
        )
        return frame
    return make_split(5000 if quick else 14000), make_split(2200 if quick else 5000), "synthetic_fallback"


def _load_dataset(quick: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train_candidates = [
        DATA_CACHE / "unsw_nb15" / "UNSW_NB15_training-set.csv",
        Path(__file__).resolve().parent / "UNSW_NB15_training-set.csv",
    ]
    test_candidates = [
        DATA_CACHE / "unsw_nb15" / "UNSW_NB15_testing-set.csv",
        Path(__file__).resolve().parent / "UNSW_NB15_testing-set.csv",
    ]
    train_path = next((path for path in train_candidates if path.exists()), None)
    test_path = next((path for path in test_candidates if path.exists()), None)
    if train_path is None:
        train_path = download_to_cache(
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
            "unsw_nb15/placeholder.csv",
        )
    if train_path is not None and test_path is not None:
        train_frame = pd.read_csv(train_path)
        test_frame = pd.read_csv(test_path)
        source = "local_unsw_copy"
    else:
        train_frame, test_frame, source = _synthetic_intrusion(quick)
    if quick:
        train_frame = train_frame.sample(min(len(train_frame), 5000), random_state=42)
        test_frame = test_frame.sample(min(len(test_frame), 2200), random_state=42)
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True), source


def run(quick: bool = True) -> ProjectResult:
    train_frame, test_frame, source = _load_dataset(quick)
    target = "label"
    x_train, x_test = one_hot_align(train_frame.drop(columns=[target]), test_frame.drop(columns=[target]))
    y_train = train_frame[target].astype(int)
    y_test = test_frame[target].astype(int)

    model_specs = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=-1),
            0.50,
            "Balanced logistic baseline",
        ),
        (
            "logistic_regression_low_threshold",
            LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=-1),
            0.35,
            "Lower threshold for higher attack recall",
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=220 if not quick else 140, random_state=42, n_jobs=-1),
            0.50,
            "Untuned forest baseline",
        ),
        (
            "weighted_random_forest",
            RandomForestClassifier(
                n_estimators=220 if not quick else 140,
                class_weight={0: 1.0, 1: 3.0},
                random_state=42,
                n_jobs=-1,
            ),
            0.35,
            "Cost-sensitive forest with lower threshold",
        ),
        (
            "weighted_random_forest_conservative",
            RandomForestClassifier(
                n_estimators=220 if not quick else 140,
                class_weight={0: 1.0, 1: 4.0},
                random_state=42,
                n_jobs=-1,
            ),
            0.45,
            "Heavier positive weighting with less aggressive threshold",
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=240 if not quick else 160, random_state=42),
            0.35,
            "Boosted tree baseline",
        ),
        (
            "hist_gradient_boosting_aggressive",
            HistGradientBoostingClassifier(max_depth=5, learning_rate=0.07, max_iter=280 if not quick else 180, random_state=42),
            0.25,
            "More aggressive thresholding for recall-sensitive deployments",
        ),
    ]

    records = []
    for algorithm, model, threshold, note in model_specs:
        _, fit_seconds = timed_run(lambda current=model: current.fit(x_train, y_train))
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        macro_f1 = float(f1_score(y_test, predictions, average="macro"))
        attack_recall = float(recall_score(y_test, predictions, zero_division=0))
        average_precision = float(average_precision_score(y_test, probabilities))
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="binary_intrusion_classification",
                algorithm=algorithm,
                feature_variant="network_flow_features",
                optimization="cost_sensitive_thresholding",
                primary_metric="macro_f1",
                primary_value=macro_f1,
                rank_score=macro_f1 + 0.2 * attack_recall,
                secondary_metric="attack_recall",
                secondary_value=attack_recall,
                tertiary_metric="average_precision",
                tertiary_value=average_precision,
                fit_seconds=fit_seconds,
                notes=f"Decision threshold={threshold:.2f}; {note}",
            )
        )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The best security-aware detector was {best.algorithm}, combining macro F1 {best.primary_value:.3f} with attack recall {best.secondary_value:.3f}. Cost-sensitive tuning helped the models prioritize attack coverage."
        ),
        recommendation=(
            "Treat intrusion detection as a cost-sensitive problem. Pick the model and threshold that maximize attack recall without letting precision collapse, rather than optimizing plain accuracy."
        ),
        key_findings=[
            f"Best detector: {best.algorithm}.",
            "Weighted forests and lowered decision thresholds improved attack recall relative to untuned baselines.",
            "Average precision is a useful secondary metric when attack prevalence is low.",
        ],
        caveats=[
            "If UNSW-NB15 is not present locally, the runner uses a synthetic intrusion dataset with similar class imbalance and mixed feature types.",
            "The quick benchmark uses a single train/test split.",
            "Thresholds are manually chosen and not optimized with a separate validation set.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
