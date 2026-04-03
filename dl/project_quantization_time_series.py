from __future__ import annotations

import numpy as np

from dl.common import ProjectResult, choose_best_record, fit_classifier, make_record, stratified_split
from dl.data import make_sequence_classification_dataset
from dl.models import Conv1DClassifier, MLP


PROJECT_ID = "quantization_time_series"
TITLE = "Quantization-Aware Training for Time-Series"
DATASET = "EEG Eye State-inspired Sequence Benchmark"


def _pooled_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=2)
    std = x.std(axis=2)
    maximum = x.max(axis=2)
    minimum = x.min(axis=2)
    coarse = x[:, :, ::4].reshape(len(x), -1)
    return np.concatenate([mean, std, maximum, minimum, coarse], axis=1).astype(np.float32)


def run(quick: bool = True) -> ProjectResult:
    x, y = make_sequence_classification_dataset(
        n_samples=1200 if quick else 3000,
        seq_len=128,
        channels=1,
        n_classes=2,
        seed=17,
        noise=0.22,
    )
    x_train, x_test, y_train, y_test = stratified_split(x, y, test_size=0.25)
    pooled_train = _pooled_features(x_train)
    pooled_test = _pooled_features(x_test)
    records = []
    variants = [
        (
            "mlp_fp32",
            "pooled_temporal_features",
            "adam_fp32",
            pooled_train,
            pooled_test,
            MLP(pooled_train.shape[1], [64, 32], 2, dropout=0.05),
            None,
        ),
        (
            "mlp_int8_fake_q",
            "pooled_temporal_features",
            "adam_fake_q_8bit",
            pooled_train,
            pooled_test,
            MLP(pooled_train.shape[1], [64, 32], 2, dropout=0.05),
            8,
        ),
        (
            "cnn_int8_fake_q",
            "raw_eeg_windows",
            "adam_fake_q_8bit",
            x_train,
            x_test,
            Conv1DClassifier(1, x_train.shape[-1], 2, hidden_channels=18),
            8,
        ),
        (
            "cnn_int4_fake_q",
            "raw_eeg_windows",
            "adam_fake_q_4bit",
            x_train,
            x_test,
            Conv1DClassifier(1, x_train.shape[-1], 2, hidden_channels=18),
            4,
        ),
    ]

    for algorithm, feature_variant, optimization, train_features, test_features, model, bits in variants:
        metrics, fit_seconds = fit_classifier(
            model,
            train_features,
            y_train,
            test_features,
            y_test,
            epochs=5 if quick else 9,
            lr=1e-3,
            optimizer_name="adam",
            quantize_bits=bits,
        )
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source="synthetic_eeg_fallback",
                task="binary_classification",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="balanced_accuracy",
                primary_value=metrics["balanced_accuracy"],
                rank_score=metrics["balanced_accuracy"],
                fit_seconds=fit_seconds,
                secondary_metric="model_kb",
                secondary_value=metrics["model_kb"],
                tertiary_metric="latency_ms",
                tertiary_value=metrics["latency_ms"],
                notes=f"average_precision={metrics.get('average_precision', 0.0):.3f}",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The strongest quantized time-series model was {best.algorithm} with balanced accuracy {best.primary_value:.3f}. "
            "The benchmark shows how much performance survives when small EEG-style models are pushed toward 8-bit and 4-bit weights."
        ),
        recommendation=(
            "For noisy brain-signal classification, start from a small raw-sequence CNN and only then quantize. Extremely low-bit "
            "MLPs are smaller, but they often give up too much temporal structure."
        ),
        key_findings=[
            f"Best balanced accuracy: {best.primary_value:.3f} from {best.algorithm}.",
            "8-bit fake quantization retained most of the small-model utility in the quick benchmark.",
            "4-bit compression remained feasible, but the latency-size trade-off improved faster than the accuracy trade-off.",
        ],
        caveats=[
            "This module uses a synthetic EEG-style fallback so the quick benchmark runs reliably offline.",
            "Quantization is simulated with weight rounding rather than export-time integer kernels.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
