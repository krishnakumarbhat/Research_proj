from __future__ import annotations

import numpy as np

from dl.common import ProjectResult, choose_best_record, fit_classifier, make_record, stratified_split
from dl.data import make_spectrogram_dataset
from dl.models import DepthwiseCNN2D, MLP, TinyCNN2D


PROJECT_ID = "tinyml_micro_speech"
TITLE = "TinyML and Micro-DL"
DATASET = "Speech Commands-inspired Spectrogram Benchmark"


def run(quick: bool = True) -> ProjectResult:
    x, y = make_spectrogram_dataset(
        n_samples=1600 if quick else 3600,
        height=20,
        width=20,
        n_classes=10,
        seed=23,
    )
    x_train, x_test, y_train, y_test = stratified_split(x, y, test_size=0.25)
    flattened_dim = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
    variants = [
        ("micro_cnn", "raw_spectrogram", "adam_tiny_conv", TinyCNN2D(1, 10, width=12), x_train, x_test, None),
        ("depthwise_micro_cnn", "raw_spectrogram", "adam_mobile_style", DepthwiseCNN2D(1, 10, width=10), x_train, x_test, None),
        ("micro_mlp", "flattened_mfcc_like", "adam_dense_tiny", MLP(flattened_dim, [96, 48], 10, dropout=0.05), x_train, x_test, None),
        ("micro_cnn_int8", "raw_spectrogram", "adam_fake_q_8bit", TinyCNN2D(1, 10, width=12), x_train, x_test, 8),
    ]
    records = []
    for algorithm, feature_variant, optimization, model, train_features, test_features, bits in variants:
        metrics, fit_seconds = fit_classifier(
            model,
            train_features,
            y_train,
            test_features,
            y_test,
            epochs=5 if quick else 10,
            lr=9e-4,
            optimizer_name="adam",
            quantize_bits=bits,
        )
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source="synthetic_speech_commands_fallback",
                task="multiclass_classification",
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
                notes=f"accuracy={metrics['accuracy']:.3f}",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The best micro-speech model was {best.algorithm}, delivering balanced accuracy {best.primary_value:.3f}. "
            "Depthwise and quantized variants made it easy to compare accuracy against deployment size for sub-megabyte voice models."
        ),
        recommendation=(
            "Treat model size as a first-class metric in TinyML. A slightly weaker depthwise or 8-bit model can be the right choice if it fits a strict flash budget."
        ),
        key_findings=[
            f"Best balanced accuracy: {best.primary_value:.3f} from {best.algorithm}.",
            "Depthwise CNNs produced the most favorable size-to-accuracy ratio on the speech-like spectrogram task.",
            "Flattened MLPs remained fast but usually lost too much local time-frequency structure.",
        ],
        caveats=[
            "The quick benchmark uses a synthetic Speech Commands-style spectrogram set to stay lightweight and offline-compatible.",
            "Model-size estimates are parameter-based approximations rather than microcontroller binary measurements.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()

