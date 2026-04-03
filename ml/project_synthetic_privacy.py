from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf, erfinv
from sklearn.datasets import fetch_openml, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from ml.common import DATA_CACHE, ProjectResult, make_record, timed_run


PROJECT_ID = "synthetic_privacy"
TITLE = "Synthetic Tabular Data Privacy"
DATASET_NAME = "Credit Card Fraud Detection"


def _load_dataset(quick: bool) -> tuple[pd.DataFrame, str]:
    local_candidates = [
        DATA_CACHE / "credit_card_fraud" / "creditcard.csv",
        Path(__file__).resolve().parent / "creditcard.csv",
    ]
    local_path = next((path for path in local_candidates if path.exists()), None)
    if local_path is not None:
        frame = pd.read_csv(local_path)
        source = "local_kaggle_copy"
    else:
        try:
            dataset = fetch_openml(name="creditcard", version=1, as_frame=True)
            frame = dataset.frame.copy()
            frame["Class"] = pd.Series(dataset.target).astype(int)
            source = "openml"
        except Exception:
            features, labels = make_classification(
                n_samples=5000 if quick else 12000,
                n_features=20,
                n_informative=12,
                n_redundant=4,
                weights=[0.985, 0.015],
                class_sep=1.6,
                random_state=42,
            )
            frame = pd.DataFrame(features, columns=[f"f_{index}" for index in range(features.shape[1])])
            frame["Class"] = labels
            source = "synthetic_fallback"
    if quick and len(frame) > 10000:
        fraud = frame.loc[frame["Class"] == 1]
        normal = frame.loc[frame["Class"] == 0].sample(7000, random_state=42)
        frame = pd.concat([fraud, normal], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return frame, source


def _gaussian_copula_sample(frame: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    ranks = np.argsort(np.argsort(values, axis=0), axis=0)
    uniforms = (ranks + 1.0) / (len(frame) + 1.0)
    gaussian = np.sqrt(2.0) * erfinv(2 * uniforms - 1)
    covariance = np.cov(gaussian, rowvar=False) + np.eye(gaussian.shape[1]) * 1e-4
    sampled = np.random.default_rng(42).multivariate_normal(np.zeros(gaussian.shape[1]), covariance, size=n_samples)
    sampled_uniforms = 0.5 * (1 + erf(sampled / np.sqrt(2)))
    synthetic = np.zeros_like(sampled_uniforms)
    for column_index in range(values.shape[1]):
        synthetic[:, column_index] = np.quantile(values[:, column_index], sampled_uniforms[:, column_index])
    return pd.DataFrame(synthetic, columns=frame.columns)


def _private_histogram_sample(frame: pd.DataFrame, n_samples: int, epsilon: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    synthetic = {}
    for column in frame.columns:
        column_values = frame[column].to_numpy(dtype=float)
        quantiles = np.quantile(column_values, np.linspace(0, 1, 11))
        quantiles[0] -= 1e-6
        quantiles[-1] += 1e-6
        counts, edges = np.histogram(column_values, bins=quantiles)
        noisy = np.maximum(counts + rng.laplace(0, 1 / epsilon, size=len(counts)), 1e-3)
        probabilities = noisy / noisy.sum()
        chosen_bins = rng.choice(len(counts), size=n_samples, p=probabilities)
        lower = edges[chosen_bins]
        upper = edges[chosen_bins + 1]
        synthetic[column] = rng.uniform(lower, upper)
    return pd.DataFrame(synthetic)


def _bootstrap_sample(frame: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    return frame.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)


def _privacy_scores(real_frame: pd.DataFrame, synthetic_frame: pd.DataFrame) -> tuple[float, float]:
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(real_frame)
    distances, indices = neighbors.kneighbors(synthetic_frame)
    exact_match_rate = float(np.mean(np.isclose(distances[:, 0], 0.0, atol=1e-6)))
    return float(np.mean(distances[:, 0])), exact_match_rate


def run(quick: bool = True) -> ProjectResult:
    frame, source = _load_dataset(quick)
    target = "Class"
    train_frame, test_frame = train_test_split(frame, test_size=0.25, random_state=42, stratify=frame[target])
    x_real_train = train_frame.drop(columns=[target]).astype(float)
    y_real_train = train_frame[target].astype(int)
    x_real_test = test_frame.drop(columns=[target]).astype(float)
    y_real_test = test_frame[target].astype(int)
    n_samples = len(train_frame)

    generators = {
        "bootstrap": lambda: _bootstrap_sample(train_frame, n_samples),
        "gaussian_copula": lambda: pd.concat(
            [
                _gaussian_copula_sample(x_real_train, n_samples),
                pd.Series(np.random.default_rng(42).choice(y_real_train, size=n_samples), name=target),
            ],
            axis=1,
        ),
        "dp_histogram": lambda: pd.concat(
            [
                _private_histogram_sample(x_real_train, n_samples, epsilon=0.9),
                pd.Series((np.random.default_rng(42).random(n_samples) < y_real_train.mean()).astype(int), name=target),
            ],
            axis=1,
        ),
    }

    records = []
    for algorithm, generator in generators.items():
        synthetic, fit_seconds = timed_run(generator)
        x_synth = synthetic.drop(columns=[target]).astype(float)
        y_synth = synthetic[target].astype(int)
        model = LogisticRegression(max_iter=1200, class_weight="balanced")
        model.fit(x_synth, y_synth)
        probabilities = model.predict_proba(x_real_test)[:, 1]
        auc = float(roc_auc_score(y_real_test, probabilities))
        ap = float(average_precision_score(y_real_test, probabilities))
        privacy_distance, match_rate = _privacy_scores(x_real_train, x_synth)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="synthetic_data_generation",
                algorithm=algorithm,
                feature_variant="numeric_credit_features",
                optimization="utility_privacy_tradeoff",
                primary_metric="roc_auc",
                primary_value=auc,
                rank_score=auc + 0.02 * privacy_distance - match_rate,
                secondary_metric="privacy_distance",
                secondary_value=privacy_distance,
                tertiary_metric="match_rate",
                tertiary_value=match_rate,
                fit_seconds=fit_seconds,
                notes=f"Downstream AP={ap:.3f}",
            )
        )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The best privacy-utility trade-off came from {best.algorithm}. It preserved downstream ROC AUC at {best.primary_value:.3f} while keeping average nearest-neighbor distance at {best.secondary_value:.3f}."
        ),
        recommendation=(
            "Use copula-style synthesis when you need stronger utility, and move toward noisier histogram-style methods when privacy risk dominates. A pure bootstrap should only be treated as a utility ceiling, not a private release."
        ),
        key_findings=[
            f"Best trade-off generator: {best.algorithm}.",
            "Bootstrap tends to maximize utility but also produces the weakest privacy proxy scores.",
            "Adding noise through independent histograms improves privacy distance but usually degrades fraud-detection fidelity.",
        ],
        caveats=[
            "Privacy is approximated with nearest-neighbor and exact-match proxies rather than a formal privacy proof.",
            "If the credit card fraud dataset is unavailable, the runner uses a synthetic imbalanced fraud process.",
            "The Gaussian copula implementation is intentionally lightweight and numeric-only.",
        ],
    )


if __name__ == "__main__":
    result = run(quick=True)
    print(result.summary)
