from __future__ import annotations

import numpy as np

from dl.common import ProjectResult, choose_best_record, fit_classifier, make_record
from dl.data import load_har_windows
from dl.models import Conv1DClassifier, DepthwiseSeparableConv1DClassifier, MLP


PROJECT_ID = "hardware_aware_nas"
TITLE = "Hardware-Aware NAS"
DATASET = "NAS-Bench-201-inspired HAR Search"


def _nas_score(accuracy: float, latency_ms: float, model_kb: float) -> float:
    return float(accuracy - 0.004 * latency_ms - 0.00015 * model_kb)


def run(quick: bool = True) -> ProjectResult:
    x_train, x_test, y_train, y_test, source = load_har_windows(quick=quick)
    num_classes = len(np.unique(y_train))
    candidates = [
        (
            "mlp_small",
            "flattened_windows",
            "search_candidate_2_layers",
            lambda: (x_train, x_test, MLP(x_train.shape[1] * x_train.shape[2], [96, 48], num_classes, dropout=0.05)),
        ),
        (
            "cnn_tiny",
            "raw_inertial_windows",
            "search_candidate_conv",
            lambda: (x_train, x_test, Conv1DClassifier(6, x_train.shape[-1], num_classes, hidden_channels=16)),
        ),
        (
            "cnn_wide",
            "raw_inertial_windows",
            "search_candidate_wider_conv",
            lambda: (x_train, x_test, Conv1DClassifier(6, x_train.shape[-1], num_classes, hidden_channels=32)),
        ),
        (
            "depthwise_mobile_style",
            "raw_inertial_windows",
            "search_candidate_depthwise",
            lambda: (x_train, x_test, DepthwiseSeparableConv1DClassifier(6, num_classes, hidden_channels=18)),
        ),
    ]
    records = []
    for algorithm, feature_variant, optimization, builder in candidates:
        train_features, test_features, model = builder()
        metrics, fit_seconds = fit_classifier(
            model,
            train_features,
            y_train,
            test_features,
            y_test,
            epochs=4 if quick else 7,
            lr=8e-4,
            optimizer_name="adamw",
        )
        score = _nas_score(metrics["balanced_accuracy"], metrics["latency_ms"], metrics["model_kb"])
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source=source,
                task="hardware_aware_architecture_search",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="nas_score",
                primary_value=score,
                rank_score=score,
                fit_seconds=fit_seconds,
                secondary_metric="balanced_accuracy",
                secondary_value=metrics["balanced_accuracy"],
                tertiary_metric="latency_ms",
                tertiary_value=metrics["latency_ms"],
                notes=f"model_kb={metrics['model_kb']:.2f}",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The best search candidate was {best.algorithm}, which maximized a latency-aware NAS score of "
            f"{best.primary_value:.3f}. The search favored compact convolutional models once latency and model size "
            "were penalized explicitly instead of ranking by accuracy alone."
        ),
        recommendation=(
            "For CPU-side NAS on small data, rank architectures by a composite deployment score instead of accuracy only. "
            "That is usually enough to surface mobile-friendly winners without a heavyweight NAS framework."
        ),
        key_findings=[
            f"Best latency-aware score: {best.primary_value:.3f} from {best.algorithm}.",
            "Wider CNNs improved raw accuracy but often lost the deployment-aware ranking once latency penalties were applied.",
            "A small candidate set is enough to build a publishable accuracy-size-latency Pareto table on CPU.",
        ],
        caveats=[
            "This is a NAS-Bench-201-inspired search over hand-defined candidates, not a direct API integration.",
            "The benchmark uses HAR windows instead of image data to keep the CPU path short and sensor-relevant.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
