from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from research_suite.paper1_visual_text.renderers import CELL_HEIGHT, CELL_WIDTH, SUPPORTED_CHARS, render_char_aware


RESAMPLING = getattr(Image, "Resampling", Image)
RESAMPLE_LANCZOS = RESAMPLING.LANCZOS


@dataclass
class OcrPrediction:
    text: str
    confidence: float


class TemplateOCR:
    def __init__(self) -> None:
        self.templates = self._build_templates()

    def _build_templates(self) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(0)
        for character in SUPPORTED_CHARS:
            image = render_char_aware(
                character,
                rng=np.random.default_rng(int(rng.integers(0, 10_000))),
                supersample=4,
                blur_radius=0.0,
                noise_level=0.0,
                jitter=0,
            )
            templates[character] = self._normalize(image)
        return templates

    def _normalize(self, image: Image.Image) -> np.ndarray:
        image = image.convert("L").resize((CELL_WIDTH, CELL_HEIGHT), resample=RESAMPLE_LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return 1.0 - array

    def recognize(self, image: Image.Image, expected_length: int) -> OcrPrediction:
        image = image.convert("L").resize(
            (max(1, expected_length * CELL_WIDTH), CELL_HEIGHT),
            resample=RESAMPLE_LANCZOS,
        )
        predicted: list[str] = []
        confidences: list[float] = []
        for index in range(expected_length):
            left = index * CELL_WIDTH
            cell = image.crop((left, 0, left + CELL_WIDTH, CELL_HEIGHT))
            cell_array = self._normalize(cell)
            ranked = []
            for character, template in self.templates.items():
                mse = float(np.mean((cell_array - template) ** 2))
                ranked.append((character, -mse))
            ranked.sort(key=lambda item: item[1], reverse=True)
            predicted.append(ranked[0][0])
            margin = ranked[0][1] - ranked[1][1]
            confidences.append(max(0.0, min(1.0, margin * 8.0)))
        return OcrPrediction(text="".join(predicted), confidence=sum(confidences) / len(confidences))
