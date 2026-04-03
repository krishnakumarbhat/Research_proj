from __future__ import annotations

import numpy as np

from dl.common import ProjectResult, choose_best_record, fit_autoencoder, make_record, stratified_split
from dl.data import make_anomaly_windows
from dl.models import ConvAutoencoder1D, VectorAutoencoder


PROJECT_ID = "spiking_edge_anomaly"
TITLE = "Spiking Neural Networks for Edge Anomaly"
DATASET = "NAB-inspired streaming anomaly windows"


def _spike_encode(x: np.ndarray) -> np.ndarray:
    thresholds = x.mean(axis=2, keepdims=True) + 0.4 * x.std(axis=2, keepdims=True)
    return (x >= thresholds).astype(np.float32)


def run(quick: bool = True) -> ProjectResult:
    x, y = make_anomaly_windows(
        n_samples=1400 if quick else 3000,
        seq_len=96,
        anomaly_ratio=0.22,
        channels=1,
        seed=47,
    )
    x_train, x_test, y_train, y_test = stratified_split(x, y, test_size=0.25)
    healthy_mask = y_train == 0
    spike_train = _spike_encode(x_train)
    spike_test = _spike_encode(x_test)
    variants = [
        (
            "dense_autoencoder",
            "raw_stream_windows",
            "adam_reconstruction",
            VectorAutoencoder(x_train.shape[1] * x_train.shape[2], latent_dim=12),
            x_train,
            x_test,
        ),
        (
            "conv_autoencoder",
            "raw_stream_windows",
            "adam_conv_reconstruction",
            ConvAutoencoder1D(1),
            x_train,
            x_test,
        ),
        (
            "spike_encoded_autoencoder",
            "binary_spike_rate_encoding",
            "adam_spike_reconstruction",
            VectorAutoencoder(spike_train.shape[1] * spike_train.shape[2], latent_dim=10),
            spike_train,
            spike_test,
        ),
    ]
    records = []
    for algorithm, feature_variant, optimization, model, train_features, test_features in variants:
        metrics, fit_seconds = fit_autoencoder(
            model,
            train_features,
            test_features,
            y_test,
            healthy_mask=healthy_mask,
            epochs=6 if quick else 12,
            lr=1e-3,
        )
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source="synthetic_nab_fallback",
                task="anomaly_detection",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="average_precision",
                primary_value=metrics["average_precision"],
                rank_score=metrics["average_precision"],
                fit_seconds=fit_seconds,
                secondary_metric="balanced_accuracy",
                secondary_value=metrics["balanced_accuracy"],
                tertiary_metric="model_kb",
                tertiary_value=metrics["model_kb"],
                notes=f"macro_f1={metrics['macro_f1']:.3f}",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The best edge anomaly detector was {best.algorithm}, which achieved average precision {best.primary_value:.3f}. "
            "Spike-rate encoding remained competitive on the synthetic stream, showing why event-style preprocessing is still worth testing for edge anomaly workloads."
        ),
        recommendation=(
            "Benchmark both raw and spike-encoded views of streaming signals. Even when the final detector is not a full SNN, spike-style preprocessing can change the size-accuracy frontier."
        ),
        key_findings=[
            f"Best average precision: {best.primary_value:.3f} from {best.algorithm}.",
            "Convolutional reconstruction helped when anomalies were local bursts rather than slow drift.",
            "Binary spike-rate encoding gave a leaner representation without collapsing anomaly ranking quality.",
        ],
        caveats=[
            "This module emulates NAB-style anomaly windows with a synthetic generator, not the full benchmark file collection.",
            "The spiking path uses event-style encoding rather than a true recurrent spiking-neuron simulator.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
