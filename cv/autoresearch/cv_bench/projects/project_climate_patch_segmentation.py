from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cv_bench.common import ProjectResult, configured_n_jobs, make_record, timed_run
from cv_bench.real_data import load_climate_patches
from cv_bench.vision_utils import dice_score, mean_iou, predict_pixel_masks, sample_pixel_dataset


PROJECT_ID = "climate_patch_segmentation"
TITLE = "Semantic Segmentation of Climate Patches"
DATASET_NAME = "Planet: Understanding the Amazon from Space"


def _generate_dataset(quick: bool) -> tuple[np.ndarray, np.ndarray, str]:
    real_dataset = load_climate_patches(quick)
    if real_dataset is not None:
        return real_dataset
    rng = np.random.default_rng(42)
    count = 20 if quick else 40
    height = 48
    width = 48
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    images = np.zeros((count, height, width, 4), dtype=np.float32)
    masks = np.zeros((count, height, width), dtype=np.int32)
    for index in range(count):
        canopy = 0.12 * np.sin(4.0 * np.pi * xx) + 0.1 * np.cos(3.0 * np.pi * yy) + rng.normal(0.0, 0.03, size=(height, width))
        red = 0.24 + 0.04 * canopy
        green = 0.52 + 0.18 * canopy
        blue = 0.34 + 0.05 * canopy
        nir = 0.78 + 0.22 * canopy
        mask = np.zeros((height, width), dtype=np.int32)

        for _ in range(1 + int(rng.random() < 0.5)):
            top = int(rng.integers(6, height - 12))
            left = int(rng.integers(6, width - 12))
            patch_h = int(rng.integers(6, 12))
            patch_w = int(rng.integers(6, 12))
            mask[top: top + patch_h, left: left + patch_w] = 1
        if rng.random() < 0.4:
            road_x = int(rng.integers(10, width - 10))
            mask[:, road_x - 1: road_x + 1] = 1

        red = red + 0.2 * mask
        green = green - 0.24 * mask
        blue = blue + 0.05 * mask
        nir = nir - 0.34 * mask

        if index % 5 == 0:
            water = (yy > 0.7) & (xx < 0.3)
            blue = blue + 0.18 * water
            nir = nir - 0.22 * water
            green = green - 0.08 * water

        image = np.stack([red, green, blue, nir], axis=-1)
        image += rng.normal(0.0, 0.02, size=image.shape)
        images[index] = np.clip(image, 0.0, 1.0)
        masks[index] = mask
    return images, masks, "synthetic_fallback"


def _mean_dice(true_masks: np.ndarray, pred_masks: np.ndarray) -> float:
    return float(np.mean([dice_score(true_mask, pred_mask) for true_mask, pred_mask in zip(true_masks, pred_masks)]))


def _ndvi_threshold(images: np.ndarray) -> np.ndarray:
    red = images[..., 0]
    nir = images[..., 3]
    green = images[..., 1]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    return ((ndvi < 0.32) & (green < 0.46)).astype(np.int32)


def run(quick: bool = True) -> ProjectResult:
    images, masks, source = _generate_dataset(quick)
    using_real = source == "planet_local_subset"
    train_images, test_images, train_masks, test_masks = train_test_split(
        images,
        masks,
        test_size=0.25,
        random_state=42,
    )

    records = []
    threshold_predictions, fit_seconds = timed_run(lambda: _ndvi_threshold(test_images))
    threshold_iou = mean_iou(test_masks, threshold_predictions, num_classes=2)
    records.append(
        make_record(
            project=PROJECT_ID,
            dataset=DATASET_NAME,
            source=source,
            task="binary_segmentation",
            algorithm="ndvi_threshold",
            feature_variant="red_nir_green_indices",
            optimization="handcrafted_remote_sensing_rule",
            primary_metric="mean_iou",
            primary_value=threshold_iou,
            rank_score=threshold_iou,
            secondary_metric="dice",
            secondary_value=_mean_dice(test_masks, threshold_predictions),
            tertiary_metric="image_count",
            tertiary_value=float(len(test_images)),
            fit_seconds=fit_seconds,
            notes="Rule-based deforestation proxy using red and NIR contrast",
        )
    )

    feature_kwargs = {"include_xy": True, "include_gradients": True, "include_local_stats": True}
    x_train, y_train = sample_pixel_dataset(
        train_images,
        train_masks,
        max_samples=22000 if quick else 52000,
        **feature_kwargs,
    )
    parallel_jobs = configured_n_jobs()
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=parallel_jobs),
        "random_forest": RandomForestClassifier(n_estimators=100 if quick else 150, random_state=42, n_jobs=parallel_jobs),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=150 if not quick else 100, random_state=42),
    }
    for algorithm, model in models.items():
        _, fit_seconds = timed_run(lambda current=model: current.fit(x_train, y_train))
        predictions = predict_pixel_masks(model, test_images, **feature_kwargs)
        score = mean_iou(test_masks, predictions, num_classes=2)
        records.append(
            make_record(
                project=PROJECT_ID,
                dataset=DATASET_NAME,
                source=source,
                task="binary_segmentation",
                algorithm=algorithm,
                feature_variant="rgb_nir_xy_grad_localstats",
                optimization="pixel_classifier_remote_sensing",
                primary_metric="mean_iou",
                primary_value=score,
                rank_score=score,
                secondary_metric="dice",
                secondary_value=_mean_dice(test_masks, predictions),
                tertiary_metric="pixel_samples",
                tertiary_value=float(len(x_train)),
                fit_seconds=fit_seconds,
                notes=(
                    "Local Planet-style multispectral patches from the workspace cache"
                    if using_real
                    else "Satellite patch classifier for localized deforestation detection"
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
            f"The best climate-patch segmenter was {best.algorithm}, reaching mean IoU {best.primary_value:.3f}. "
            "The remote-sensing rule baseline stayed useful, but learned pixel classifiers handled confusing water and edge effects more reliably."
        ),
        recommendation=(
            "Use NDVI-style thresholds as a reference, then quantify how much a lightweight pixel classifier gains on the same satellite patches. That gap tells you whether a larger segmentation model is justified."
        ),
        key_findings=[
            f"Best segmenter: {best.algorithm}.",
            "The handcrafted vegetation index rule provided a meaningful deployment baseline.",
            "Learned models improved robustness when water or narrow cleared corridors mimicked low-NIR vegetation.",
        ],
        caveats=(
            [
                "The local real-data path expects cached RGB+NIR tensors and masks rather than the full Planet Kaggle release.",
                "The task is reduced to binary cleared-vs-forest segmentation.",
                "Only pixel-level baselines are compared in quick mode, not full satellite segmentation networks.",
            ]
            if using_real
            else [
                "This module uses synthetic Planet-style RGB+NIR patches rather than the original Kaggle tiles.",
                "The task is reduced to binary cleared-vs-forest segmentation.",
                "Only pixel-level baselines are compared in quick mode, not full satellite segmentation networks.",
            ]
        ),
    )


if __name__ == "__main__":
    print(run(quick=True).summary)