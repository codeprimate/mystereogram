"""Utility helpers for device selection and image preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_device(device_override: Optional[str] = None) -> str:
    """Return the preferred device string, honoring a valid override."""
    # Lazy import to avoid delay on module import
    import torch
    
    if device_override is not None:
        device = device_override.strip().lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device override: {device_override}")
        return device

    # Default to CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def validate_image_format(image_path: str | Path) -> bool:
    """Return True if the path has a supported image file extension."""
    path = Path(image_path)
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk and return it as RGB."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    try:
        image = Image.open(path)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or invalid image file: {path}") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_image(image: Image.Image) -> Image.Image:
    """Resize an image to exactly 1MP (1,000,000 pixels) while maintaining aspect ratio."""
    width, height = image.size
    current_pixels = width * height
    target_pixels = 1_000_000

    # Calculate scale factor to reach exactly 1MP
    scale = math.sqrt(target_pixels / current_pixels)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)
