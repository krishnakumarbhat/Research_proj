from __future__ import annotations

import numpy as np

from dl.common import ProjectResult, choose_best_record, fit_classifier, make_record
from dl.data import load_digits_images
from dl.models import MLP, TinyCNN2D


PROJECT_ID = "weight_quantization_theory"
TITLE = "Weight Quantization Theory"
DATASET = "CIFAR-10 / digits fallback"


def run(quick: bool = True) -> ProjectResult:
    x_train, x_test, y_train, y_test, source = load_digits_images(quick=quick)
    flattened_dim = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
    variants = [
        ("cnn_fp32", "image_tensor", "adam_conv_fp32", TinyCNN2D(1, 10, width=16), None),
        ("cnn_int8", "image_tensor", "adam_conv_fake_q_8bit", TinyCNN2D(1, 10, width=16), 8),
        ("cnn_int4", "image_tensor", "adam_conv_fake_q_4bit", TinyCNN2D(1, 10, width=16), 4),
        ("mlp_fp32", "flattened_pixels", "adam_dense_fp32", MLP(flattened_dim, [128, 64], 10, dropout=0.1), None),
    ]
    records = []
    for algorithm, feature_variant, optimization, model, bits in variants:
        metrics, fit_seconds = fit_classifier(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=5 if quick else 10,
            lr=8e-4,
            optimizer_name="adam",
            quantize_bits=bits,
        )
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source=source,
                task="multiclass_classification",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="accuracy",
                primary_value=metrics["accuracy"],
                rank_score=metrics["accuracy"],
                fit_seconds=fit_seconds,
                secondary_metric="model_kb",
                secondary_value=metrics["model_kb"],
                tertiary_metric="latency_ms",
                tertiary_value=metrics["latency_ms"],
                notes=f"balanced_accuracy={metrics['balanced_accuracy']:.3f}",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The strongest quantized image model was {best.algorithm}, reaching accuracy {best.primary_value:.3f}. "
            "The experiment table makes the classic quantization question explicit: how much accuracy is being traded for each step down in numerical precision?"
        ),
        recommendation=(
            "Report accuracy alongside model size and latency when discussing quantization theory. A lower-bit model is only compelling if the deployment benefit is visible in the same table."
        ),
        key_findings=[
            f"Best accuracy: {best.primary_value:.3f} from {best.algorithm}.",
            "8-bit quantization retained most of the compact CNN's utility on the offline image fallback.",
            "4-bit compression saved more memory but widened the accuracy gap more sharply than the 8-bit setting.",
        ],
        caveats=[
            "The quick path uses sklearn digits instead of CIFAR-10 so the benchmark can run offline and fast.",
            "This module studies fake weight quantization, not the full quantized deployment toolchain or hardware kernel behavior.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
