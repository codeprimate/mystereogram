"""Utility helpers for device selection and image preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError


# Common image file extensions (for quick validation)
# Note: PIL supports many more formats, so we also try to open files
# to see if PIL can handle them
COMMON_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    ".ico", ".pcx", ".ppm", ".pgm", ".pbm", ".xbm", ".xpm", ".svg",
    ".heic", ".heif", ".avif", ".jp2", ".j2k", ".jpf", ".jpx",
}


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
    """
    Validate that an image file can be opened by PIL.
    
    This function tries to open the file with PIL to see if it's a supported
    image format. PIL supports many formats beyond just checking extensions.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if the file appears to be a valid image format that PIL can open
    """
    path = Path(image_path)
    
    # Quick check: if extension is in common list, likely valid
    if path.suffix.lower() in COMMON_IMAGE_EXTENSIONS:
        return True
    
    # Try to open the file to see if PIL can handle it
    # This catches formats PIL supports that aren't in our extension list
    try:
        with Image.open(path) as img:
            # Verify it's actually an image by checking format
            return img.format is not None
    except (UnidentifiedImageError, OSError, IOError):
        return False


def normalize_image(image: Image.Image) -> Image.Image:
    """
    Normalize an image to RGB format, handling various formats and containers.
    
    Handles:
    - RGBA/LA: Composits on white background to remove transparency
    - Palette (P): Converts palette to RGB
    - Grayscale (L): Converts to RGB
    - CMYK: Converts to RGB
    - Other modes: Converts to RGB
    
    Args:
        image: PIL Image in any format
        
    Returns:
        PIL Image in RGB mode
    """
    if image.mode == "RGB":
        return image
    
    # Handle palette images (may have transparency)
    if image.mode == "P":
        if "transparency" in image.info:
            # Convert to RGBA to preserve transparency
            image = image.convert("RGBA")
        else:
            # No transparency, convert directly to RGB
            return image.convert("RGB")
    
    # Handle transparency by compositing on white background
    if image.mode in ("RGBA", "LA"):
        # Create white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        # Composite the image on white background using alpha channel as mask
        background.paste(image, mask=image.split()[-1])
        return background
    
    # Handle other modes by direct conversion
    if image.mode in ("L", "1", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"):
        return image.convert("RGB")
    
    # Fallback: try to convert to RGB
    return image.convert("RGB")


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image from disk and return it as RGB.
    
    Supports all image formats that PIL/Pillow can handle, including:
    - Raster formats: JPEG, PNG, GIF, WebP, BMP, TIFF, ICO, PCX, etc.
    - Some vector formats: SVG (if pillow-simd or similar plugin installed)
    - Modern formats: HEIC, HEIF, AVIF, JPEG 2000 (if plugins installed)
    
    For animated formats (GIF, WebP), loads the first frame.
    For multi-page formats (TIFF), loads the first page.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image in RGB mode
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be opened as an image
        OSError: If there's an I/O error reading the file
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    try:
        # Open the image - PIL will handle format detection
        image = Image.open(path)
        
        # For animated formats (GIF, WebP), get the first frame
        # For multi-page formats (TIFF), get the first page
        if hasattr(image, "is_animated") and image.is_animated:
            # Seek to first frame (frame 0)
            image.seek(0)
        elif hasattr(image, "n_frames") and image.n_frames > 1:
            # Multi-page format (like TIFF), get first page
            image.seek(0)
        
        # Load the image data into memory
        # This is important for formats that use lazy loading
        image.load()
        
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"Unsupported or invalid image file: {path}. "
            "The file format may not be supported by PIL/Pillow."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Error reading image file: {path}. "
            "The file may be corrupted or in an unsupported format."
        ) from exc

    return normalize_image(image)


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
