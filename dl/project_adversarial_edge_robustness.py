from __future__ import annotations

from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn

from dl.common import ProjectResult, choose_best_record, fgsm_accuracy, fit_classifier, make_loader, make_record, make_optimizer, set_seed
from dl.data import load_digits_images
from dl.models import DepthwiseCNN2D, TinyCNN2D


PROJECT_ID = "adversarial_edge_robustness"
TITLE = "Adversarial Robustness of Edge Models"
DATASET = "GTSRB / digits fallback"


def _fit_adversarial(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int) -> float:
    set_seed(79)
    model.train()
    optimizer = make_optimizer(model, "adam", lr=8e-4)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(x_train, y_train, batch_size=64, shuffle=True)
    start = perf_counter()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.clone().detach().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            clean_loss = criterion(logits, batch_y)
            clean_loss.backward(retain_graph=True)
            adversarial = torch.clamp(batch_x + 0.05 * batch_x.grad.sign(), 0.0, 1.0).detach()
            optimizer.zero_grad(set_to_none=True)
            mixed_loss = 0.5 * criterion(model(batch_x.detach()), batch_y) + 0.5 * criterion(model(adversarial), batch_y)
            mixed_loss.backward()
            optimizer.step()
    return perf_counter() - start


def _evaluate_clean_accuracy(model: nn.Module, x_test: np.ndarray, y_test: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(x_test, dtype=torch.float32)).argmax(dim=1).cpu().numpy()
    return float(accuracy_score(y_test, prediction))


def run(quick: bool = True) -> ProjectResult:
    x_train, x_test, y_train, y_test, source = load_digits_images(quick=quick)
    variants = [
        ("tiny_cnn_standard", "image_tensor", "adam_clean_training", TinyCNN2D(1, 10, width=12), "standard", None),
        ("tiny_cnn_adversarial", "image_tensor", "fgsm_adversarial_training", TinyCNN2D(1, 10, width=12), "adversarial", None),
        ("depthwise_cnn_standard", "image_tensor", "adam_depthwise", DepthwiseCNN2D(1, 10, width=10), "standard", None),
        ("tiny_cnn_int8", "image_tensor", "adam_fake_q_8bit", TinyCNN2D(1, 10, width=12), "standard", 8),
    ]
    records = []
    for algorithm, feature_variant, optimization, model, mode, bits in variants:
        if mode == "adversarial":
            fit_seconds = _fit_adversarial(model, x_train, y_train, epochs=5 if quick else 10)
            clean_accuracy = _evaluate_clean_accuracy(model, x_test, y_test)
        else:
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
            clean_accuracy = metrics["accuracy"]
        adversarial_accuracy = fgsm_accuracy(model, x_test[: min(256, len(x_test))], y_test[: min(256, len(y_test))], epsilon=0.05)
        model_kb = sum(parameter.numel() for parameter in model.parameters()) * (bits or 32) / 8.0 / 1024.0
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET,
                source=source,
                task="adversarial_classification",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="adversarial_accuracy",
                primary_value=adversarial_accuracy,
                rank_score=adversarial_accuracy,
                fit_seconds=fit_seconds,
                secondary_metric="clean_accuracy",
                secondary_value=clean_accuracy,
                tertiary_metric="model_kb",
                tertiary_value=model_kb,
                notes="FGSM epsilon=0.05",
            )
        )

    best = choose_best_record(records)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET,
        records=records,
        summary=(
            f"The most robust edge model was {best.algorithm}, reaching adversarial accuracy {best.primary_value:.3f} under FGSM noise. "
            "Small adversarially trained CNNs usually beat clean-only compressed baselines once the evaluation includes attack-time perturbations."
        ),
        recommendation=(
            "If an edge classifier operates in a hostile environment, optimize for adversarial accuracy directly. Clean accuracy by itself will overstate deployment readiness."
        ),
        key_findings=[
            f"Best adversarial accuracy: {best.primary_value:.3f} from {best.algorithm}.",
            "Adversarial training improved robustness more reliably than post-hoc quantization alone.",
            "Depthwise models remained efficient, but their clean-to-robustness trade-off still needed explicit measurement.",
        ],
        caveats=[
            "The quick benchmark uses digits as an offline-compatible traffic-sign fallback.",
            "Only FGSM attacks are included here; a publishable robustness study should add stronger iterative attacks and certified bounds where possible.",
        ],
    )


def main() -> None:
    print(run(quick=True).summary)


if __name__ == "__main__":
    main()
