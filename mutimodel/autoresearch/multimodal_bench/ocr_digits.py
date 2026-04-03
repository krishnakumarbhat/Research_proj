from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .common import (
    RANDOM_STATE,
    ProjectResult,
    classification_metrics,
    choose_best_record,
    make_record,
    timed_run,
)
from .datasets import load_digit_images


PROJECT_ID = "image_to_text"
TITLE = "Image-to-Text (OCR proxy)"
DATASET_NAME = "sklearn digits"


def _hog_features(images: np.ndarray) -> np.ndarray:
    return np.vstack(
        [
            hog(
                image,
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(1, 1),
                feature_vector=True,
            )
            for image in images
        ]
    ).astype(np.float32)


@dataclass(slots=True)
class DigitOCRModel:
    algorithm: str
    model: object

    def predict_images(self, images: np.ndarray) -> np.ndarray:
        if self.algorithm == "hog_linear_svm":
            return self.model.predict(_hog_features(images))
        flat = images.reshape(len(images), -1)
        return self.model.predict(flat)


def _fit_knn(images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray) -> DigitOCRModel:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3)),
        ]
    )
    model.fit(images.reshape(len(images), -1)[train_idx], labels[train_idx])
    return DigitOCRModel("template_knn", model)


def _fit_hog_svm(images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray) -> DigitOCRModel:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=2.0, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(_hog_features(images[train_idx]), labels[train_idx])
    return DigitOCRModel("hog_linear_svm", model)


def _fit_mlp(images: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, quick: bool) -> DigitOCRModel:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=160 if quick else 260,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(images.reshape(len(images), -1)[train_idx], labels[train_idx])
    return DigitOCRModel("mlp_pixels", model)


def build_reference_model(images: np.ndarray, labels: np.ndarray, preferred_algorithm: str, quick: bool = False) -> DigitOCRModel:
    indices = np.arange(len(images))
    if preferred_algorithm == "template_knn":
        return _fit_knn(images, labels, indices)
    if preferred_algorithm == "hog_linear_svm":
        return _fit_hog_svm(images, labels, indices)
    if preferred_algorithm == "mlp_pixels":
        return _fit_mlp(images, labels, indices, quick)
    raise ValueError(f"Unknown OCR algorithm: {preferred_algorithm}")


def run(quick: bool = False) -> tuple[ProjectResult, DigitOCRModel]:
    images, labels, source = load_digit_images()
    train_idx, test_idx = train_test_split(
        np.arange(len(images)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    records = []
    algorithms = [
        (
            "template_knn",
            lambda: _fit_knn(images, labels, train_idx),
            "raw_pixels",
            "nearest_neighbor_template_matching",
        ),
        (
            "hog_linear_svm",
            lambda: _fit_hog_svm(images, labels, train_idx),
            "hog_descriptors",
            "linear_margin_classifier",
        ),
        (
            "mlp_pixels",
            lambda: _fit_mlp(images, labels, train_idx, quick),
            "raw_pixels",
            "two_layer_mlp",
        ),
    ]

    for algorithm, fit_fn, feature_variant, optimization in algorithms:
        bundle, fit_seconds = timed_run(fit_fn)
        predictions = bundle.predict_images(images[test_idx])
        metrics = classification_metrics(labels[test_idx], predictions)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="ocr_digit_reading",
                algorithm=algorithm,
                feature_variant=feature_variant,
                optimization=optimization,
                primary_metric="accuracy",
                primary_value=metrics["accuracy"],
                rank_score=metrics["accuracy"],
                secondary_metric="balanced_accuracy",
                secondary_value=metrics["balanced_accuracy"],
                tertiary_metric="macro_f1",
                tertiary_value=metrics["f1"],
                fit_seconds=fit_seconds,
                notes=f"{len(train_idx)} train / {len(test_idx)} test images",
            )
        )

    best = choose_best_record(records)
    reference_model = build_reference_model(images, labels, best.algorithm, quick=quick)
    summary = (
        f"The strongest OCR proxy was {best.algorithm} with accuracy {best.primary_value:.3f}. "
        "Template methods remain surprisingly competitive on tiny clean digits, but handcrafted HOG "
        "or learned pixel classifiers usually produce the most stable results."
    )
    recommendation = (
        "For small OCR tasks, use a feature-engineered HOG or learned pixel baseline before escalating to a "
        "larger vision-language stack."
    )
    key_findings = [
        f"Best held-out accuracy was {best.primary_value:.3f} from {best.algorithm}.",
        "A k-NN template reader is a meaningful mechanical-era baseline on clean digits.",
        "HOG plus a linear margin model remains a strong feature-engineering baseline.",
    ]
    caveats = [
        "This benchmark uses clean 8x8 digit images rather than messy scanned documents.",
        "The task maps digits to token labels, not rich document understanding or layout extraction.",
        "A single random split is used instead of repeated cross-validation.",
    ]
    return (
        ProjectResult(
            project=PROJECT_ID,
            title=TITLE,
            dataset=DATASET_NAME,
            source=source,
            task="image_to_text",
            records=records,
            summary=summary,
            recommendation=recommendation,
            key_findings=key_findings,
            caveats=caveats,
            artifacts=[],
        ),
        reference_model,
    )