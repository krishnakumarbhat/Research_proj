from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import f1_score


def as_image_batch(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ..., None]
    elif array.ndim == 3:
        array = array[..., None]
    if array.max(initial=0.0) > 1.5:
        array = array / 255.0
    return array


def stack_pixel_features(
    images: np.ndarray,
    *,
    include_xy: bool = True,
    include_gradients: bool = True,
    include_local_stats: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    batch = as_image_batch(images)
    num_images, height, width, channels = batch.shape
    features = [batch.reshape(num_images * height * width, channels)]

    if include_xy:
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, height, dtype=np.float32),
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        coords = np.stack([yy, xx], axis=-1)
        coords = np.broadcast_to(coords, (num_images, height, width, 2))
        features.append(coords.reshape(num_images * height * width, 2))

    if include_gradients:
        grad_chunks = []
        for channel in range(channels):
            channel_view = batch[..., channel]
            grad_y = np.gradient(channel_view, axis=1)
            grad_x = np.gradient(channel_view, axis=2)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_chunks.extend([grad_y[..., None], grad_x[..., None], grad_mag[..., None]])
        gradients = np.concatenate(grad_chunks, axis=-1)
        features.append(gradients.reshape(num_images * height * width, gradients.shape[-1]))

    if include_local_stats:
        local_mean = uniform_filter(batch, size=(1, 3, 3, 1), mode="nearest")
        local_var = uniform_filter(batch**2, size=(1, 3, 3, 1), mode="nearest") - local_mean**2
        local_std = np.sqrt(np.clip(local_var, 0.0, None))
        stats = np.concatenate([local_mean, local_std], axis=-1)
        features.append(stats.reshape(num_images * height * width, stats.shape[-1]))

    return np.concatenate(features, axis=1), (num_images, height, width)


def sample_pixel_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    max_samples: int,
    random_state: int = 42,
    **feature_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    features, _ = stack_pixel_features(images, **feature_kwargs)
    labels = np.asarray(masks).reshape(-1)
    if len(labels) <= max_samples:
        return features, labels

    rng = np.random.default_rng(random_state)
    unique = np.unique(labels)
    chosen = []
    per_class = max(1, max_samples // max(len(unique), 1))
    for label in unique:
        label_indices = np.flatnonzero(labels == label)
        take = min(len(label_indices), per_class)
        chosen.append(rng.choice(label_indices, size=take, replace=False))
    stacked = np.concatenate(chosen)
    if len(stacked) < max_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), stacked, assume_unique=False)
        top_up = rng.choice(remaining, size=min(len(remaining), max_samples - len(stacked)), replace=False)
        stacked = np.concatenate([stacked, top_up])
    rng.shuffle(stacked)
    return features[stacked], labels[stacked]


def predict_pixel_masks(model: Any, images: np.ndarray, **feature_kwargs: Any) -> np.ndarray:
    features, (num_images, height, width) = stack_pixel_features(images, **feature_kwargs)
    predictions = model.predict(features)
    return np.asarray(predictions).reshape(num_images, height, width)


def mean_iou(true_masks: np.ndarray, pred_masks: np.ndarray, num_classes: int | None = None) -> float:
    truth = np.asarray(true_masks)
    pred = np.asarray(pred_masks)
    if num_classes is None:
        num_classes = int(max(truth.max(initial=0), pred.max(initial=0)) + 1)

    ious = []
    for label in range(num_classes):
        truth_label = truth == label
        pred_label = pred == label
        union = np.logical_or(truth_label, pred_label).sum()
        if union == 0:
            continue
        intersection = np.logical_and(truth_label, pred_label).sum()
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def dice_score(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    truth = np.asarray(true_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)
    intersection = np.logical_and(truth, pred).sum()
    denom = truth.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * intersection) / denom)


def binary_f1(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    return float(f1_score(np.asarray(true_mask).reshape(-1), np.asarray(pred_mask).reshape(-1), zero_division=0))


def box_iou(box_a: np.ndarray | tuple[float, float, float, float], box_b: np.ndarray | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def center_error(box_a: np.ndarray | tuple[float, float, float, float], box_b: np.ndarray | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    center_a = np.asarray([(ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0])
    center_b = np.asarray([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
    return float(np.linalg.norm(center_a - center_b))


def patch_descriptors(
    images: np.ndarray,
    *,
    patch_size: int,
    stride: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    batch = as_image_batch(images)
    descriptors: list[np.ndarray] = []
    mapping: list[tuple[int, int, int]] = []
    for image_index in range(batch.shape[0]):
        image = batch[image_index]
        gray = image.mean(axis=-1)
        grad_y = np.gradient(gray, axis=0)
        grad_x = np.gradient(gray, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        for top in range(0, image.shape[0] - patch_size + 1, stride):
            for left in range(0, image.shape[1] - patch_size + 1, stride):
                patch = image[top: top + patch_size, left: left + patch_size]
                grad_patch = grad_mag[top: top + patch_size, left: left + patch_size]
                descriptor = np.concatenate(
                    [
                        patch.mean(axis=(0, 1)),
                        patch.std(axis=(0, 1)),
                        np.asarray([grad_patch.mean(), grad_patch.std(), top / image.shape[0], left / image.shape[1]], dtype=np.float32),
                    ]
                )
                descriptors.append(descriptor.astype(np.float32))
                mapping.append((image_index, top, left))
    return np.asarray(descriptors, dtype=np.float32), mapping


def patch_scores_to_maps(
    scores: np.ndarray,
    mapping: list[tuple[int, int, int]],
    *,
    image_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    height, width = image_shape
    image_count = max((image_index for image_index, _, _ in mapping), default=-1) + 1
    score_maps = np.zeros((image_count, height, width), dtype=np.float32)
    counts = np.zeros((image_count, height, width), dtype=np.float32)
    for score, (image_index, top, left) in zip(scores, mapping):
        score_maps[image_index, top: top + patch_size, left: left + patch_size] += float(score)
        counts[image_index, top: top + patch_size, left: left + patch_size] += 1.0
    counts[counts == 0] = 1.0
    return score_maps / counts