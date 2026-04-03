from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from research_suite.common.io_utils import ensure_directory, write_csv, write_json
from research_suite.common.metrics import cer, mean, wer
from research_suite.paper1_visual_text.ocr import TemplateOCR
from research_suite.paper1_visual_text.renderers import (
    render_baseline_texture,
    render_char_aware,
    render_layout_guided,
)


PHRASES = [
    "LAUNCH TUESDAY",
    "BIO SENSOR 7",
    "NOVA LAB",
    "VISION STACK",
    "CLIMATE REPORT",
    "ROBOTICS REVIEW",
    "EDGE CACHE",
    "GRAPH MEMORY",
    "CHUNKING STUDY",
    "OCR REWARD LOOP",
    "GLYPH CONTROL",
    "TEXT DIFFUSER",
    "MARIO BENCH",
    "WORD ERROR RATE",
    "CHAR ERROR RATE",
    "LAYOUT PRIOR",
    "BYTE LEVEL TOKEN",
    "HIGH FIDELITY",
    "LATENT NOISE",
    "CURATED DATASET",
]


def reward_guided_render(
    text: str,
    ocr: TemplateOCR,
    rng: np.random.Generator,
    supersample: int,
    candidates: int,
    blur_radius: float,
    noise_level: float,
) -> tuple[object, float]:
    best_image = None
    best_score = -1.0
    for _ in range(candidates):
        candidate_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        image = render_char_aware(
            text,
            rng=candidate_rng,
            supersample=supersample,
            blur_radius=blur_radius,
            noise_level=noise_level,
            jitter=1,
        )
        prediction = ocr.recognize(image, expected_length=len(text))
        score = (1.0 - cer(text, prediction.text)) + (0.1 * prediction.confidence)
        if score > best_score:
            best_image = image
            best_score = score
    return best_image, best_score


def build_configs() -> list[dict[str, object]]:
    return [
        {
            "config_id": "baseline_ds6_noise006",
            "algorithm": "baseline_texture",
            "downsample_factor": 6,
            "blur_radius": 1.2,
            "noise_level": 0.06,
            "jitter": 2,
        },
        {
            "config_id": "baseline_ds8_noise010",
            "algorithm": "baseline_texture",
            "downsample_factor": 8,
            "blur_radius": 1.5,
            "noise_level": 0.10,
            "jitter": 3,
        },
        {
            "config_id": "baseline_ds10_noise012",
            "algorithm": "baseline_texture",
            "downsample_factor": 10,
            "blur_radius": 1.8,
            "noise_level": 0.12,
            "jitter": 3,
        },
        {
            "config_id": "layout_mild",
            "algorithm": "layout_guided",
            "blur_radius": 0.4,
            "noise_level": 0.02,
            "jitter": 1,
        },
        {
            "config_id": "layout_balanced",
            "algorithm": "layout_guided",
            "blur_radius": 0.7,
            "noise_level": 0.03,
            "jitter": 1,
        },
        {
            "config_id": "layout_robust",
            "algorithm": "layout_guided",
            "blur_radius": 0.9,
            "noise_level": 0.04,
            "jitter": 1,
        },
        {
            "config_id": "charaware_x3",
            "algorithm": "char_aware",
            "supersample": 3,
            "blur_radius": 0.2,
            "noise_level": 0.01,
            "jitter": 0,
        },
        {
            "config_id": "charaware_x4",
            "algorithm": "char_aware",
            "supersample": 4,
            "blur_radius": 0.1,
            "noise_level": 0.01,
            "jitter": 0,
        },
        {
            "config_id": "ocrreward_c3",
            "algorithm": "ocr_rewarded",
            "supersample": 3,
            "blur_radius": 0.2,
            "noise_level": 0.02,
            "candidates": 3,
        },
        {
            "config_id": "ocrreward_c6",
            "algorithm": "ocr_rewarded",
            "supersample": 4,
            "blur_radius": 0.2,
            "noise_level": 0.02,
            "candidates": 6,
        },
    ]


def run(output_root: str | Path, quick: bool = False, seed: int = 42) -> dict[str, object]:
    output_dir = ensure_directory(Path(output_root) / "paper1_visual_text")
    sample_dir = ensure_directory(output_dir / "samples")
    ocr = TemplateOCR()
    phrases = PHRASES[:8] if quick else PHRASES
    full_rows: list[dict[str, object]] = []

    for config_index, config in enumerate(build_configs()):
        config_id = str(config["config_id"])
        for phrase_index, phrase in enumerate(phrases):
            rng = np.random.default_rng(seed + (config_index * 10_000) + phrase_index)
            started = perf_counter()
            reward_score = None
            if config["algorithm"] == "baseline_texture":
                image = render_baseline_texture(
                    phrase,
                    rng=rng,
                    downsample_factor=int(config["downsample_factor"]),
                    blur_radius=float(config["blur_radius"]),
                    noise_level=float(config["noise_level"]),
                    jitter=int(config["jitter"]),
                )
            elif config["algorithm"] == "layout_guided":
                image = render_layout_guided(
                    phrase,
                    rng=rng,
                    blur_radius=float(config["blur_radius"]),
                    noise_level=float(config["noise_level"]),
                    jitter=int(config["jitter"]),
                )
            elif config["algorithm"] == "char_aware":
                image = render_char_aware(
                    phrase,
                    rng=rng,
                    supersample=int(config["supersample"]),
                    blur_radius=float(config["blur_radius"]),
                    noise_level=float(config["noise_level"]),
                    jitter=int(config["jitter"]),
                )
            else:
                image, reward_score = reward_guided_render(
                    phrase,
                    ocr=ocr,
                    rng=rng,
                    supersample=int(config["supersample"]),
                    candidates=int(config["candidates"]),
                    blur_radius=float(config["blur_radius"]),
                    noise_level=float(config["noise_level"]),
                )
            runtime_ms = (perf_counter() - started) * 1000.0
            prediction = ocr.recognize(image, expected_length=len(phrase))
            if phrase_index < 2:
                image.save(sample_dir / f"{config_id}_{phrase_index}.png")
            full_rows.append(
                {
                    "paper": "paper1",
                    "config_id": config_id,
                    "algorithm": config["algorithm"],
                    "phrase": phrase,
                    "prediction": prediction.text,
                    "cer": round(cer(phrase, prediction.text), 6),
                    "wer": round(wer(phrase, prediction.text), 6),
                    "exact_match": int(prediction.text == phrase),
                    "confidence": round(prediction.confidence, 6),
                    "reward_score": round(reward_score, 6) if reward_score is not None else "",
                    "runtime_ms": round(runtime_ms, 4),
                }
            )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in full_rows:
        grouped[str(row["config_id"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for config in build_configs():
        config_id = str(config["config_id"])
        rows = grouped[config_id]
        summary_rows.append(
            {
                "paper": "paper1",
                "config_id": config_id,
                "algorithm": config["algorithm"],
                "mean_cer": round(mean([float(row["cer"]) for row in rows]), 6),
                "mean_wer": round(mean([float(row["wer"]) for row in rows]), 6),
                "exact_match_rate": round(mean([float(row["exact_match"]) for row in rows]), 6),
                "mean_confidence": round(mean([float(row["confidence"]) for row in rows]), 6),
                "mean_runtime_ms": round(mean([float(row["runtime_ms"]) for row in rows]), 4),
                "details": {key: value for key, value in config.items() if key not in {"config_id", "algorithm"}},
            }
        )

    summary_rows.sort(key=lambda row: (row["mean_cer"], -row["exact_match_rate"], row["mean_runtime_ms"]))
    write_csv(output_dir / "full_results.csv", full_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_json(
        output_dir / "metadata.json",
        {
            "paper": "paper1",
            "quick": quick,
            "seed": seed,
            "phrases": phrases,
            "best_config": summary_rows[0],
        },
    )
    return {
        "paper": "paper1",
        "summary_rows": summary_rows,
        "full_rows": full_rows,
        "best_config": summary_rows[0],
        "sample_dir": str(sample_dir),
    }