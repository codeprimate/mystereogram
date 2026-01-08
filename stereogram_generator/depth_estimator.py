"""DepthAnything model loading utilities."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

# Suppress warnings for better UX
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .utils import get_device, load_image, resize_image


DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
# Set to True if model output needs inversion for "closer = higher".
INVERT_DEPTH = False
# Default padding size in pixels for depth map
DEFAULT_PADDING = 0


def load_depth_model(
    device: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_ID,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
) -> Tuple["AutoModelForDepthEstimation", "AutoImageProcessor"]:
    """Load DepthAnything model and processor for depth estimation."""
    # Lazy import to avoid delay on module import
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    
    selected_device = device or get_device()
    resolved_cache_dir = str(cache_dir) if cache_dir is not None else None

    try:
        processor = AutoImageProcessor.from_pretrained(
            model_path,
            cache_dir=resolved_cache_dir,
            local_files_only=local_files_only,
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            model_path,
            cache_dir=resolved_cache_dir,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise OSError(f"Failed to load depth model from {model_path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error loading model {model_path}: {exc}") from exc

    model.to(selected_device)
    model.eval()
    return model, processor


def preprocess_image_for_depth(
    image: Image.Image,
    processor: "AutoImageProcessor",
    device: str,
) -> "torch.Tensor":
    """Prepare image tensor for depth model inference."""
    # Lazy import to avoid delay on module import
    import torch
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    return pixel_values


def _estimate_depth(
    image: Image.Image,
    model: "AutoModelForDepthEstimation",
    processor: "AutoImageProcessor",
    device: str,
) -> "torch.Tensor":
    """Run depth inference and return the predicted depth tensor (internal helper)."""
    # Lazy import to avoid delay on module import
    import torch
    
    pixel_values = preprocess_image_for_depth(image, processor, device)
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs.predicted_depth


def interpolate_depth_map(
    depth_tensor: "torch.Tensor",
    target_size: tuple[int, int],
) -> "torch.Tensor":
    """Interpolate depth tensor to the target (height, width)."""
    # Lazy import to avoid delay on module import
    import torch.nn.functional as F
    
    if depth_tensor.dim() == 2:
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    elif depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(1)
    elif depth_tensor.dim() != 4:
        raise ValueError("Depth tensor must be 2D, 3D, or 4D.")

    interpolated = F.interpolate(
        depth_tensor,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )
    return interpolated.squeeze(1)


def normalize_depth_map(depth_array: np.ndarray) -> np.ndarray:
    """Normalize depth values to [0, 1] range with min-max scaling."""
    depth_min = float(np.min(depth_array))
    depth_max = float(np.max(depth_array))
    if math.isclose(depth_max, depth_min):
        return np.zeros_like(depth_array, dtype=np.float32)
    normalized = (depth_array - depth_min) / (depth_max - depth_min)
    return normalized.astype(np.float32)


def invert_depth_map(depth_array: np.ndarray) -> np.ndarray:
    """Invert normalized depth values (1.0 - depth)."""
    return 1.0 - depth_array


def pad_depth_map(depth_array: np.ndarray, padding: int = DEFAULT_PADDING) -> np.ndarray:
    """
    Add white padding around a depth map.
    
    Args:
        depth_array: 2D normalized depth map array with values in [0, 1]
        padding: Padding size in pixels (default: 50)
    
    Returns:
        Padded depth map with white (1.0) padding around the edges
    """
    if depth_array.ndim != 2:
        raise ValueError("Depth map must be a 2D array.")
    if padding < 0:
        raise ValueError("Padding must be non-negative.")
    
    height, width = depth_array.shape
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    
    # Create padded array filled with white (1.0 for maximum depth)
    padded = np.ones((padded_height, padded_width), dtype=depth_array.dtype)
    
    # Place original depth map in the center
    padded[padding:padding + height, padding:padding + width] = depth_array
    
    return padded


def postprocess_depth_map(
    depth_tensor: "torch.Tensor",
    original_size: tuple[int, int],
) -> np.ndarray:
    """Resize, normalize, and return depth map as a numpy array."""
    target_size = (original_size[1], original_size[0])
    interpolated = interpolate_depth_map(depth_tensor, target_size=target_size)
    depth_array = interpolated.squeeze().cpu().numpy()
    return normalize_depth_map(depth_array)


def estimate_depth_map(
    image_path: str | Path,
    device: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    model: Optional["AutoModelForDepthEstimation"] = None,
    processor: Optional["AutoImageProcessor"] = None,
    model_path: str = DEFAULT_MODEL_ID,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    console: Optional["Console"] = None,
    progress: Optional["Progress"] = None,
    padding: int = DEFAULT_PADDING,
) -> np.ndarray:
    """Estimate and optionally save a normalized depth map for an image."""
    if console is None:
        from rich.console import Console
        console = Console()
    
    if progress:
        task1 = progress.add_task("[cyan]Loading image...", total=None)
    
    image = load_image(image_path)
    resized = resize_image(image)
    resized_size = resized.size
    
    if progress:
        progress.update(task1, completed=True)
        task2 = progress.add_task("[cyan]Loading depth estimation model...", total=None)

    selected_device = device or get_device()
    if model is None or processor is None:
        model, processor = load_depth_model(
            device=selected_device,
            model_path=model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    
    if progress:
        progress.update(task2, completed=True)
        task3 = progress.add_task("[cyan]Generating depth map...", total=None)

    depth_tensor = _estimate_depth(resized, model, processor, selected_device)
    depth_map = postprocess_depth_map(depth_tensor, resized_size)
    if INVERT_DEPTH:
        depth_map = invert_depth_map(depth_map)
    
    # Add white padding around the depth map
    depth_map = pad_depth_map(depth_map, padding=padding)
    
    if progress:
        progress.update(task3, completed=True)

    if save_path is not None:
        save_path = Path(save_path)
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            depth_image = Image.fromarray((depth_map * 255).astype(np.uint8))
            depth_image.save(save_path)
        except OSError as exc:
            raise OSError(f"Failed to save depth map to {save_path}: {exc}") from exc

    return depth_map
