import numpy as np

from research_suite.paper1_visual_text.ocr import TemplateOCR
from research_suite.paper1_visual_text.renderers import (
    render_baseline_texture,
    render_char_aware,
    render_layout_guided,
)
from research_suite.common.metrics import cer


def test_template_ocr_recognizes_clean_render() -> None:
    ocr = TemplateOCR()
    image = render_char_aware(
        "NOVA LAB",
        rng=np.random.default_rng(7),
        supersample=4,
        blur_radius=0.0,
        noise_level=0.0,
        jitter=0,
    )
    prediction = ocr.recognize(image, expected_length=len("NOVA LAB"))
    assert prediction.text == "NOVA LAB"


def test_baseline_produces_high_cer() -> None:
    ocr = TemplateOCR()
    image = render_baseline_texture(
        "HELLO",
        rng=np.random.default_rng(42),
        downsample_factor=8,
        blur_radius=1.5,
        noise_level=0.10,
        jitter=3,
    )
    prediction = ocr.recognize(image, expected_length=5)
    assert cer("HELLO", prediction.text) > 0.3


def test_layout_guided_produces_low_cer() -> None:
    ocr = TemplateOCR()
    image = render_layout_guided(
        "EDGE CACHE",
        rng=np.random.default_rng(99),
        blur_radius=0.4,
        noise_level=0.02,
        jitter=1,
    )
    prediction = ocr.recognize(image, expected_length=len("EDGE CACHE"))
    assert cer("EDGE CACHE", prediction.text) == 0.0


def test_char_aware_with_digits() -> None:
    ocr = TemplateOCR()
    image = render_char_aware(
        "BIO SENSOR 7",
        rng=np.random.default_rng(13),
        supersample=4,
        blur_radius=0.1,
        noise_level=0.01,
        jitter=0,
    )
    prediction = ocr.recognize(image, expected_length=len("BIO SENSOR 7"))
    assert prediction.text == "BIO SENSOR 7"
