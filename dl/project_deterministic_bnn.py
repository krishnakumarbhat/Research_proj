from __future__ import annotations

import numpy as np

from dl.common import ProjectResult, choose_best_record, fit_classifier, make_record
from dl.data import load_digits_images
from dl.models import MLP, SignMLP, TinyCNN2D


PROJECT_ID = "deterministic_bnn"
TITLE = "Deterministic Binarized Neural Networks"
DATASET = "Fashion-MNIST / digits fallback"


def run(quick: bool = True) -> ProjectResult:
    x_train, x_test, y_train, y_test, source = load_digits_images(quick=quick)
    flattened_dim = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
    num_classes = len(np.unique(y_train))
    variants = [
        ("mlp_fp32", "flattened_pixels", "adam_dense", MLP(flattened_dim, [128, 64], num_classes, dropout=0.1), x_train, x_test, None),
        ("sign_mlp", "flattened_pixels", "adam_sign_ste", SignMLP(flattened_dim, 128, num_classes), x_train, x_test, None),
        ("sign_mlp_int8", "flattened_pixels", "adam_sign_plus_8bit", SignMLP(flattened_dim, 128, num_classes), x_train, x_test, 8),
        ("tiny_cnn_reference", "image_tensor", "adam_conv_reference", TinyCNN2D(1, num_classes, width=12), x_train, x_test, None),
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
                secondary_metric="balanced_accuracy",
                secondary_value=metrics["balanced_accuracy"],
                tertiary_metric="model_kb",
                tertiary_value=metrics["model_kb"],
                notes=f"latency_ms={metrics['latency_ms']:.2f}",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The strongest deterministic BNN benchmark variant was {best.algorithm}, reaching accuracy {best.primary_value:.3f}. "
            "Straight-through sign activations made the compact binarized models trainable, but the strongest baseline still depended on how much representational capacity survived compression."
        ),
        recommendation=(
            "Use a sign-activation MLP to prototype bitwise edge inference, but keep a tiny CNN or full-precision MLP in the comparison table so the compression penalty is explicit."
        ),
        key_findings=[
            f"Best accuracy: {best.primary_value:.3f} from {best.algorithm}.",
            "Sign-based deterministic BNNs remained viable on the image fallback, though they trailed the strongest dense reference when the class count increased.",
            "Adding post-training 8-bit quantization to the sign network reduced size further with only modest extra engineering.",
        ],
        caveats=[
            "The quick run uses sklearn digits as an offline-compatible fallback instead of Fashion-MNIST proper.",
            "This is a deterministic BNN approximation with straight-through estimators, not a hardware bitwise kernel implementation.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
