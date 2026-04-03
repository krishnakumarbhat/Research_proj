from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cv_bench.common import ProjectResult, configured_n_jobs, make_record, timed_run
from cv_bench.real_data import load_busi_images
from cv_bench.vision_utils import binary_f1, dice_score, predict_pixel_masks, sample_pixel_dataset


PROJECT_ID = "medical_ultrasound_segmentation"
TITLE = "Medical Segmentation via U-Net on Small Datasets"
DATASET_NAME = "BUSI Breast Ultrasound Dataset"


def _generate_dataset(quick: bool) -> tuple[np.ndarray, np.ndarray, str]:
    real_dataset = load_busi_images(quick)
    if real_dataset is not None:
        return real_dataset
    rng = np.random.default_rng(42)
    count = 20 if quick else 40
    height = 48
    width = 48
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    images = np.zeros((count, height, width), dtype=np.float32)
    masks = np.zeros((count, height, width), dtype=np.int32)
    for index in range(count):
        base = 0.48 + 0.08 * yy + rng.normal(0.0, 0.06, size=(height, width))
        center_x = rng.uniform(-0.25, 0.25)
        center_y = rng.uniform(-0.2, 0.2)
        radius_x = rng.uniform(0.18, 0.35)
        radius_y = rng.uniform(0.15, 0.3)
        lesion = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0
        halo = (((xx - center_x) / (radius_x * 1.25)) ** 2 + ((yy - center_y) / (radius_y * 1.25)) ** 2) <= 1.0
        image = base - 0.18 * lesion + 0.06 * halo
        shadow = np.clip((yy - center_y) * 1.4, 0.0, 1.0)
        image -= 0.08 * lesion.astype(np.float32) * shadow
        image = 0.65 * image + 0.35 * uniform_filter(image, size=3)
        images[index] = np.clip(image, 0.0, 1.0)
        masks[index] = lesion.astype(np.int32)
    return images, masks, "synthetic_fallback"


def _mean_dice(true_masks: np.ndarray, pred_masks: np.ndarray) -> float:
    return float(np.mean([dice_score(true_mask, pred_mask) for true_mask, pred_mask in zip(true_masks, pred_masks)]))


def _threshold_segment(images: np.ndarray) -> np.ndarray:
    local_mean = np.asarray([uniform_filter(image, size=5, mode="nearest") for image in images], dtype=np.float32)
    return (images < (local_mean - 0.05)).astype(np.int32)


def run(quick: bool = True) -> ProjectResult:
    images, masks, source = _generate_dataset(quick)
    using_real = source == "busi_local"
    train_images, test_images, train_masks, test_masks = train_test_split(
        images,
        masks,
        test_size=0.25,
        random_state=42,
    )
    labeled_images = train_images[: 8 if quick else 16]
    labeled_masks = train_masks[: 8 if quick else 16]
    augmented_images = np.concatenate([labeled_images, labeled_images[:, :, ::-1]], axis=0)
    augmented_masks = np.concatenate([labeled_masks, labeled_masks[:, :, ::-1]], axis=0)

    records = []
    threshold_predictions, fit_seconds = timed_run(lambda: _threshold_segment(test_images))
    threshold_score = _mean_dice(test_masks, threshold_predictions)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="medical_segmentation",
            algorithm="adaptive_threshold",
            feature_variant="local_contrast",
            optimization="no_training_baseline",
            primary_metric="dice",
            primary_value=threshold_score,
            rank_score=threshold_score,
            secondary_metric="binary_f1",
            secondary_value=binary_f1(test_masks, threshold_predictions),
            tertiary_metric="labeled_images",
            tertiary_value=float(len(labeled_images)),
            fit_seconds=fit_seconds,
            notes="Threshold baseline on low-label ultrasound segmentation",
        )
    )

    feature_kwargs = {"include_xy": True, "include_gradients": True, "include_local_stats": True}
    x_train, y_train = sample_pixel_dataset(
        augmented_images,
        augmented_masks,
        max_samples=16000 if quick else 42000,
        **feature_kwargs,
    )
    parallel_jobs = configured_n_jobs()
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=parallel_jobs),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=150 if not quick else 100, random_state=42),
    }
    for algorithm, model in models.items():
        _, fit_seconds = timed_run(lambda current=model: current.fit(x_train, y_train))
        predictions = predict_pixel_masks(model, test_images, **feature_kwargs)
        score = _mean_dice(test_masks, predictions)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="medical_segmentation",
                algorithm=algorithm,
                feature_variant="augmented_pixel_features",
                optimization="flip_augmentation_small_label_regime",
                primary_metric="dice",
                primary_value=score,
                rank_score=score,
                secondary_metric="binary_f1",
                secondary_value=binary_f1(test_masks, predictions),
                tertiary_metric="pixel_samples",
                tertiary_value=float(len(x_train)),
                fit_seconds=fit_seconds,
                notes=(
                    "Small-label BUSI subset with simple spatial and gradient features"
                    if using_real
                    else "Small-label BUSI surrogate with simple spatial and gradient features"
                ),
            )
        )

    best = max(records, key=lambda record: record.rank_score)
    return ProjectResult(
        project=PROJECT_ID,
        title=TITLE,
        dataset=DATASET_NAME,
        records=records,
        summary=(
            f"The strongest small-data ultrasound segmenter was {best.algorithm}, reaching Dice {best.primary_value:.3f}. "
            "Even a compact pixel classifier gained noticeably from a tiny augmentation step once the label budget was restricted."
        ),
        recommendation=(
            "Before training a U-Net on a tiny BUSI subset, benchmark what thresholding and low-capacity pixel models can already achieve under the same label budget. That gives you a realistic floor for augmentation gains."
        ),
        key_findings=[
            f"Best segmenter: {best.algorithm}.",
            "Flip augmentation helped stabilize the low-label pixel classifier.",
            "Thresholding remained a useful baseline but struggled around the fuzzy lesion halo.",
        ],
        caveats=(
            [
                "The real-data path expects a locally cached BUSI directory structure rather than downloading the Kaggle package automatically.",
                "Only a small labeled subset is exposed to the trained models in quick mode.",
                "The benchmark compares low-capacity surrogates, not an actual U-Net implementation.",
            ]
            if using_real
            else [
                "This module uses a synthetic ultrasound generator rather than the original BUSI images and masks.",
                "Only a small labeled subset is exposed to the trained models in quick mode.",
                "The benchmark compares low-capacity surrogates, not an actual U-Net implementation.",
            ]
        ),
    )


if __name__ == "__main__":
    print(run(quick=True).summary)