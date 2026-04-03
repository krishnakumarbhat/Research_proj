from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dl.common import DATA_CACHE, RANDOM_STATE, download_to_cache, maybe_downsample, set_seed


HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def _har_root() -> Path | None:
    zip_path = download_to_cache(HAR_URL, "uci_har/UCI_HAR_Dataset.zip")
    if zip_path is None:
        return None
    extract_dir = DATA_CACHE / "uci_har" / "extracted"
    dataset_root = extract_dir / "UCI HAR Dataset"
    if dataset_root.exists():
        return dataset_root
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return dataset_root if dataset_root.exists() else None


def load_har_windows(quick: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    root = _har_root()
    signal_names = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
    ]
    if root is None:
        x, y = make_sequence_classification_dataset(
            n_samples=1800 if quick else 4200,
            seq_len=128,
            channels=6,
            n_classes=6,
            seed=RANDOM_STATE,
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        return x_train, x_test, y_train, y_test, "synthetic_har_fallback"

    def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
        arrays = []
        for name in signal_names:
            path = root / split / "Inertial Signals" / f"{name}_{split}.txt"
            arrays.append(np.loadtxt(path, dtype=np.float32))
        x_split = np.stack(arrays, axis=1)
        y_split = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
        return x_split, y_split

    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    if quick:
        x_train, y_train = maybe_downsample(x_train, y_train, max_rows=1600)
        x_test, y_test = maybe_downsample(x_test, y_test, max_rows=700)
    return x_train, x_test, y_train, y_test, "uci_har"


def make_sequence_classification_dataset(
    *,
    n_samples: int,
    seq_len: int,
    channels: int,
    n_classes: int,
    seed: int,
    noise: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    patterns = np.zeros((n_classes, channels, seq_len), dtype=np.float32)
    for cls in range(n_classes):
        for channel in range(channels):
            freq = 1 + cls + (channel % 3)
            phase = (channel + 1) * 0.3 + cls * 0.2
            base = np.sin(2 * np.pi * freq * t + phase) + 0.35 * np.cos(np.pi * (channel + 1) * t)
            if cls % 2 == 0:
                base += 0.2 * np.sign(np.sin(2 * np.pi * (cls + 1) * t))
            patterns[cls, channel] = base.astype(np.float32)

    labels = rng.integers(0, n_classes, size=n_samples)
    samples = np.zeros((n_samples, channels, seq_len), dtype=np.float32)
    for index, label in enumerate(labels):
        amplitude = rng.uniform(0.8, 1.25)
        shift = rng.normal(0.0, 0.03, size=(channels, 1)).astype(np.float32)
        samples[index] = amplitude * patterns[label] + shift + rng.normal(0.0, noise, size=(channels, seq_len))
    return samples, labels.astype(np.int64)


def make_spectrogram_dataset(
    *,
    n_samples: int,
    height: int,
    width: int,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing="ij")
    samples = np.zeros((n_samples, 1, height, width), dtype=np.float32)
    labels = rng.integers(0, n_classes, size=n_samples)
    for index, label in enumerate(labels):
        ridge = np.exp(-((grid_y - np.sin((label + 1) * grid_x)) ** 2) / (0.08 + 0.02 * label))
        harmonics = np.cos((label + 1) * np.pi * grid_x) * np.sin((label + 2) * np.pi * grid_y)
        samples[index, 0] = ridge + 0.4 * harmonics + rng.normal(0.0, 0.08, size=(height, width))
    samples -= samples.min()
    samples /= samples.max() + 1e-6
    return samples.astype(np.float32), labels.astype(np.int64)


def load_digits_images(quick: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)
    x = images[:, None, :, :]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    if quick:
        x_train, y_train = maybe_downsample(x_train, y_train, max_rows=1100)
        x_test, y_test = maybe_downsample(x_test, y_test, max_rows=450)
    return x_train, x_test, y_train, y_test, "sklearn_digits_fallback"


def make_text_sentiment_dataset(n_samples: int = 1400, seed: int = RANDOM_STATE) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = ["great", "excellent", "fun", "moving", "sharp", "love", "brilliant", "warm"]
    negative = ["awful", "flat", "boring", "grim", "waste", "hate", "weak", "messy"]
    neutral = ["movie", "plot", "cast", "music", "ending", "scene", "dialogue", "visuals"]
    texts: list[str] = []
    labels: list[int] = []
    for _ in range(n_samples):
        label = int(rng.integers(0, 2))
        anchor = positive if label == 1 else negative
        tokens = rng.choice(anchor, size=3, replace=True).tolist()
        tokens += rng.choice(neutral, size=4, replace=True).tolist()
        if rng.random() < 0.25:
            tokens.append(rng.choice(negative if label == 1 else positive))
        rng.shuffle(tokens)
        texts.append(" ".join(tokens))
        labels.append(label)
    return texts, np.array(labels, dtype=np.int64)


def make_stock_regression_dataset(
    *,
    n_steps: int,
    window: int,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prices = [100.0]
    regime = 0.002
    for index in range(1, n_steps + window + 1):
        if index % 120 == 0:
            regime *= -1.0
        drift = regime + 0.0008 * np.sin(index / 12)
        shock = rng.normal(0.0, 0.015)
        prices.append(prices[-1] * (1.0 + drift + shock))
    prices = np.array(prices, dtype=np.float32)
    windows = []
    targets = []
    for start in range(n_steps):
        segment = prices[start : start + window]
        features = np.stack(
            [
                segment / (segment[0] + 1e-6),
                np.gradient(segment),
                np.log(segment + 1e-6),
            ],
            axis=0,
        )
        windows.append(features)
        targets.append((prices[start + window] - prices[start + window - 1]) / prices[start + window - 1])
    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


def make_ecg_dataset(
    *,
    n_samples: int,
    seq_len: int,
    n_classes: int,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    templates = []
    for cls in range(n_classes):
        qrs_center = 0.2 + cls * 0.08
        qrs = np.exp(-((t - qrs_center) ** 2) / (0.0015 + 0.0004 * cls))
        p_wave = 0.35 * np.exp(-((t - 0.1) ** 2) / 0.006)
        t_wave = 0.4 * np.exp(-((t - 0.65) ** 2) / (0.01 + 0.002 * cls))
        baseline = 0.15 * np.sin(2 * np.pi * (cls + 1) * t)
        templates.append((p_wave + qrs + t_wave + baseline).astype(np.float32))
    labels = rng.integers(0, n_classes, size=n_samples)
    samples = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    for index, label in enumerate(labels):
        stretch = rng.uniform(0.95, 1.05)
        warped_t = np.clip(np.linspace(0.0, 1.0, seq_len) * stretch, 0.0, 1.0)
        samples[index, 0] = np.interp(warped_t, t, templates[label]) + rng.normal(0.0, 0.05, size=seq_len)
    return samples, labels.astype(np.int64)


def make_anomaly_windows(
    *,
    n_samples: int,
    seq_len: int,
    anomaly_ratio: float,
    channels: int = 1,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    x = np.zeros((n_samples, channels, seq_len), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    anomaly_count = int(n_samples * anomaly_ratio)
    anomaly_indices = set(rng.choice(n_samples, size=anomaly_count, replace=False).tolist())
    for index in range(n_samples):
        base = np.sin(2 * np.pi * 3 * t) + 0.4 * np.cos(2 * np.pi * 7 * t)
        sample = np.stack([base + 0.05 * channel for channel in range(channels)], axis=0)
        sample += rng.normal(0.0, 0.04, size=sample.shape)
        if index in anomaly_indices:
            y[index] = 1
            spike_index = rng.integers(seq_len // 5, seq_len - seq_len // 5)
            sample[:, spike_index : spike_index + 3] += rng.uniform(1.5, 2.5)
            sample += 0.4 * np.sign(np.sin(16 * np.pi * t))[None, :]
        x[index] = sample
    return x, y


def make_tabular_drift_dataset(
    *,
    n_samples: int,
    n_features: int,
    n_segments: int,
    n_classes: int,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    segment_ids = np.repeat(np.arange(n_segments), n_samples // n_segments)
    if len(segment_ids) < n_samples:
        segment_ids = np.concatenate([segment_ids, np.full(n_samples - len(segment_ids), n_segments - 1)])
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for segment in range(n_segments):
        weights = rng.normal(0.0, 1.0, size=(n_classes, n_features))
        weights += segment * 0.35
        mask = segment_ids == segment
        logits = x[mask] @ weights.T
        y[mask] = logits.argmax(axis=1)
        x[mask, :4] += segment * 0.6
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32)
    return x, y, segment_ids.astype(np.int64)


def make_burgers_dataset(
    *,
    n_x: int = 96,
    n_t: int = 60,
    viscosity: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n_x, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_t, dtype=np.float32)
    dx = x[1] - x[0]
    outer_dt = t[1] - t[0]
    substeps = 12
    dt = outer_dt / substeps
    u = np.zeros((n_t, n_x), dtype=np.float32)
    u[0] = -np.sin(np.pi * x)
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    current = u[0].copy()
    for time_index in range(0, n_t - 1):
        for _ in range(substeps):
            next_state = current.copy()
            convection = current[1:-1] * (current[1:-1] - current[:-2]) / dx
            diffusion = viscosity * (current[2:] - 2 * current[1:-1] + current[:-2]) / (dx ** 2)
            next_state[1:-1] = current[1:-1] - dt * convection + dt * diffusion
            next_state[0] = 0.0
            next_state[-1] = 0.0
            current = np.clip(next_state, -2.0, 2.0)
        u[time_index + 1] = current
    grid_t, grid_x = np.meshgrid(t, x, indexing="ij")
    coords = np.stack([grid_x.reshape(-1), grid_t.reshape(-1)], axis=1).astype(np.float32)
    values = u.reshape(-1).astype(np.float32)
    return coords, values, u
