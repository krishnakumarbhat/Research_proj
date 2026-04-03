from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


SUPPORTED_CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/:&.,"
CELL_WIDTH = 24
CELL_HEIGHT = 40
RESAMPLING = getattr(Image, "Resampling", Image)
RESAMPLE_BILINEAR = RESAMPLING.BILINEAR
RESAMPLE_LANCZOS = RESAMPLING.LANCZOS
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_text_grid(text: str, scale: int, jitter: int, rng: np.random.Generator) -> Image.Image:
    width = max(1, len(text) * CELL_WIDTH * scale)
    height = CELL_HEIGHT * scale
    cell_width = CELL_WIDTH * scale
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    font = load_font(max(10, int(height * 0.72)))
    for index, character in enumerate(text):
        left = index * cell_width
        if character == " ":
            continue
        bbox = draw.textbbox((0, 0), character, font=font)
        glyph_width = bbox[2] - bbox[0]
        glyph_height = bbox[3] - bbox[1]
        offset_x = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        offset_y = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        x = left + (cell_width - glyph_width) / 2 - bbox[0] + offset_x
        y = (height - glyph_height) / 2 - bbox[1] + offset_y
        draw.text((x, y), character, fill=0, font=font)
    return image


def finalize_image(image: Image.Image, noise_level: float, blur_radius: float, rng: np.random.Generator) -> Image.Image:
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if noise_level > 0:
        array = np.asarray(image, dtype=np.float32)
        array += rng.normal(0.0, 255.0 * noise_level, size=array.shape)
        array = np.clip(array, 0.0, 255.0)
        image = Image.fromarray(array.astype(np.uint8), mode="L")
    return image


def render_baseline_texture(
    text: str,
    rng: np.random.Generator,
    downsample_factor: int,
    blur_radius: float,
    noise_level: float,
    jitter: int,
) -> Image.Image:
    image = render_text_grid(text, scale=2, jitter=jitter, rng=rng)
    low_width = max(len(text), image.width // max(2, downsample_factor))
    low_height = max(8, image.height // max(2, downsample_factor))
    image = image.resize((low_width, low_height), resample=RESAMPLE_BILINEAR)
    image = image.resize((len(text) * CELL_WIDTH, CELL_HEIGHT), resample=RESAMPLE_BILINEAR)
    image = finalize_image(image, noise_level=noise_level, blur_radius=blur_radius, rng=rng)
    image = ImageEnhance.Contrast(image).enhance(0.82)
    return image


def render_layout_guided(
    text: str,
    rng: np.random.Generator,
    blur_radius: float,
    noise_level: float,
    jitter: int,
) -> Image.Image:
    image = render_text_grid(text, scale=3, jitter=jitter, rng=rng)
    image = image.resize((len(text) * CELL_WIDTH, CELL_HEIGHT), resample=RESAMPLE_LANCZOS)
    image = finalize_image(image, noise_level=noise_level, blur_radius=blur_radius, rng=rng)
    image = ImageEnhance.Sharpness(image).enhance(1.2)
    return image


def render_char_aware(
    text: str,
    rng: np.random.Generator,
    supersample: int,
    blur_radius: float,
    noise_level: float,
    jitter: int,
) -> Image.Image:
    image = render_text_grid(text, scale=supersample, jitter=jitter, rng=rng)
    image = image.resize((len(text) * CELL_WIDTH, CELL_HEIGHT), resample=RESAMPLE_LANCZOS)
    image = finalize_image(image, noise_level=noise_level, blur_radius=blur_radius, rng=rng)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return image
