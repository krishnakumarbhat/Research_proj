from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, cast
import zipfile

import numpy as np
import requests
from sklearn.datasets import load_digits

from .common import DATA_CACHE, RANDOM_STATE, ensure_directories


FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
DIGIT_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]


@dataclass(slots=True)
class AudioExample:
    label: int
    text: str
    audio: np.ndarray
    sr: int
    speaker: str
    source: str


def digit_name(label: int) -> str:
    return DIGIT_NAMES[int(label)]


def load_digit_images() -> tuple[np.ndarray, np.ndarray, str]:
    digits = cast(Any, load_digits(return_X_y=False))
    images = (digits.images.astype(np.float32) / 16.0).clip(0.0, 1.0)
    labels = digits.target.astype(int)
    return images, labels, "sklearn_digits"


def _fsdd_recordings_dir() -> Path:
    return DATA_CACHE / "fsdd" / "free-spoken-digit-dataset-master" / "recordings"


def _download_fsdd() -> Path | None:
    ensure_directories()
    recordings_dir = _fsdd_recordings_dir()
    if recordings_dir.exists():
        return recordings_dir

    archive_dir = DATA_CACHE / "fsdd"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "master.zip"

    try:
        response = requests.get(FSDD_URL, timeout=90)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(archive_dir)
    except requests.RequestException:
        return None

    return recordings_dir if recordings_dir.exists() else None


def _synthetic_audio(label: int, index: int, sr: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE + label * 1000 + index)
    duration = rng.uniform(0.35, 0.75)
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    base = 170.0 + label * 22.0
    harmonic = base * (1.4 + 0.03 * label)
    chirp = base + (25.0 + 2.5 * label) * t
    envelope = np.exp(-2.2 * np.linspace(0.0, 1.0, t.size))
    signal = 0.55 * np.sin(2.0 * np.pi * base * t)
    signal += 0.25 * np.sin(2.0 * np.pi * harmonic * t)
    signal += 0.12 * np.sin(2.0 * np.pi * chirp * t)
    signal *= 0.8 + 0.2 * np.sin(2.0 * np.pi * (1.5 + 0.1 * label) * t)
    if label % 2 == 0:
        burst = np.zeros_like(signal)
        start = int(0.1 * sr)
        stop = min(signal.size, start + int(0.08 * sr))
        burst[start:stop] = 0.2 * np.sin(2.0 * np.pi * (base * 2.0) * t[: stop - start])
        signal += burst
    signal = envelope * signal + rng.normal(0.0, 0.01, size=signal.size)
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)


def _generate_synthetic_spoken_digits(max_per_digit: int | None, sr: int) -> list[AudioExample]:
    count = max_per_digit or 48
    examples: list[AudioExample] = []
    for label in range(10):
        for index in range(count):
            examples.append(
                AudioExample(
                    label=label,
                    text=digit_name(label),
                    audio=_synthetic_audio(label, index, sr),
                    sr=sr,
                    speaker=f"synthetic_{index % 4}",
                    source="synthetic_spoken_digits",
                )
            )
    return examples


def load_spoken_digits(
    max_per_digit: int | None = None,
    sr: int = 8000,
    prefer_real: bool = True,
) -> tuple[list[AudioExample], str]:
    try:
        librosa = importlib.import_module("librosa")
    except ImportError:
        librosa = None

    if librosa is None:
        examples = _generate_synthetic_spoken_digits(max_per_digit, sr)
        return examples, "synthetic_spoken_digits"

    recordings_dir = _download_fsdd() if prefer_real else None
    if recordings_dir is None:
        examples = _generate_synthetic_spoken_digits(max_per_digit, sr)
        return examples, "synthetic_spoken_digits"

    counts: dict[int, int] = defaultdict(int)
    examples: list[AudioExample] = []
    for wav_path in sorted(recordings_dir.glob("*.wav")):
        label_text, speaker, _ = wav_path.stem.split("_", 2)
        label = int(label_text)
        if max_per_digit is not None and counts[label] >= max_per_digit:
            continue
        audio, _ = librosa.load(wav_path, sr=sr)
        audio = audio.astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        examples.append(
            AudioExample(
                label=label,
                text=digit_name(label),
                audio=audio,
                sr=sr,
                speaker=speaker,
                source="free_spoken_digit_dataset",
            )
        )
        counts[label] += 1
    return examples, "free_spoken_digit_dataset"
