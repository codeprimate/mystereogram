"""Noise-based autostereogram generation utilities."""

from __future__ import annotations

import numpy as np
import noise

# Pattern generation constants
DEFAULT_PATTERN_TYPE = "perlin"
DEFAULT_PERLIN_SCALE = 0.2
DEFAULT_PERLIN_OCTAVES = 6

# HSV color control defaults
DEFAULT_HUE_RANGE = (0.0, 1.0)  # Full spectrum (0-360° in normalized 0-1)
DEFAULT_SATURATION_RANGE = (0.7, 1.0)  # High saturation for vibrant colors
DEFAULT_VALUE_RANGE = (0.7, 1.0)  # High brightness for vivid colors

# Noise width calculation constants
NOISE_WIDTH_MIN = 64
NOISE_WIDTH_MAX = 128
NOISE_WIDTH_DIVISOR = 15

# Shift range calculation constants
SHIFT_RANGE_MIN = 20
SHIFT_RANGE_MAX = 60
SHIFT_RANGE_DIVISOR = 30


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert HSV color space to RGB.
    
    Args:
        h: Hue array in range [0, 1] (0=red, 1/3=green, 2/3=blue)
        s: Saturation array in range [0, 1]
        v: Value (brightness) array in range [0, 1]
    
    Returns:
        RGB array with shape (..., 3) and values in range [0, 255] as uint8
    """
    # Ensure inputs are numpy arrays
    h = np.asarray(h)
    s = np.asarray(s)
    v = np.asarray(v)
    
    # Clamp values to valid ranges
    h = np.clip(h, 0.0, 1.0)
    s = np.clip(s, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Convert hue to 0-6 range for easier calculation
    h6 = h * 6.0
    
    # Calculate sector (0-5) and fractional part
    sector = h6.astype(np.int32) % 6
    f = h6 - sector
    
    # Calculate intermediate values
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    # Initialize RGB arrays
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    
    # Map sector to RGB
    mask0 = sector == 0
    r[mask0] = v[mask0]
    g[mask0] = t[mask0]
    b[mask0] = p[mask0]
    
    mask1 = sector == 1
    r[mask1] = q[mask1]
    g[mask1] = v[mask1]
    b[mask1] = p[mask1]
    
    mask2 = sector == 2
    r[mask2] = p[mask2]
    g[mask2] = v[mask2]
    b[mask2] = t[mask2]
    
    mask3 = sector == 3
    r[mask3] = p[mask3]
    g[mask3] = q[mask3]
    b[mask3] = v[mask3]
    
    mask4 = sector == 4
    r[mask4] = t[mask4]
    g[mask4] = p[mask4]
    b[mask4] = v[mask4]
    
    mask5 = sector == 5
    r[mask5] = v[mask5]
    g[mask5] = p[mask5]
    b[mask5] = q[mask5]
    
    # Stack and convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).clip(0, 255).astype(np.uint8)


def generate_static_pattern(width: int, height: int, seed: int | None = None) -> np.ndarray:
    """Generate a random grayscale noise pattern."""
    if width <= 0 or height <= 0:
        raise ValueError("Noise pattern dimensions must be positive.")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def generate_perlin_pattern(
    width: int,
    height: int,
    seed: int | None = None,
    scale: float = DEFAULT_PERLIN_SCALE,
    octaves: int = DEFAULT_PERLIN_OCTAVES,
    color: bool = True,
    hue_range: tuple[float, float] = DEFAULT_HUE_RANGE,
    saturation_range: tuple[float, float] = DEFAULT_SATURATION_RANGE,
    value_range: tuple[float, float] = DEFAULT_VALUE_RANGE,
    hue_scale: float | None = None,
    saturation_scale: float | None = None,
    value_scale: float | None = None,
) -> np.ndarray:
    """
    Generate a Perlin noise pattern for smooth wave-like appearance.
    
    Args:
        width: Pattern width in pixels
        height: Pattern height in pixels
        seed: Random seed for reproducibility
        scale: Perlin noise scale factor (lower = smoother)
        octaves: Number of noise octaves
        color: If True, generate RGB color pattern using HSV space
        hue_range: (min, max) hue range in [0, 1]
        saturation_range: (min, max) saturation range in [0, 1]
        value_range: (min, max) value range in [0, 1]
        hue_scale: Separate scale for hue noise (defaults to scale)
        saturation_scale: Separate scale for saturation noise (defaults to scale)
        value_scale: Separate scale for value noise (defaults to scale)
    
    Returns:
        Grayscale pattern (height, width) or RGB pattern (height, width, 3) as uint8
    """
    if width <= 0 or height <= 0:
        raise ValueError("Noise pattern dimensions must be positive.")
    if scale <= 0:
        raise ValueError("Perlin scale must be positive.")
    if octaves < 1:
        raise ValueError("Perlin octaves must be at least 1.")
    if hue_range[0] < 0 or hue_range[1] > 1 or hue_range[0] >= hue_range[1]:
        raise ValueError("Hue range must be in [0, 1] with min < max.")
    if saturation_range[0] < 0 or saturation_range[1] > 1 or saturation_range[0] >= saturation_range[1]:
        raise ValueError("Saturation range must be in [0, 1] with min < max.")
    if value_range[0] < 0 or value_range[1] > 1 or value_range[0] >= value_range[1]:
        raise ValueError("Value range must be in [0, 1] with min < max.")

    # Use seed as base offset if provided, otherwise use 0
    base = int(seed) if seed is not None else 0
    
    # Use separate scales for each channel, defaulting to main scale
    h_scale = hue_scale if hue_scale is not None else scale
    s_scale = saturation_scale if saturation_scale is not None else scale
    v_scale = value_scale if value_scale is not None else scale
    
    if h_scale <= 0 or s_scale <= 0 or v_scale <= 0:
        raise ValueError("All Perlin scales must be positive.")

    if not color:
        # Generate grayscale Perlin noise
        pattern = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                noise_value = noise.pnoise2(
                    x * scale,
                    y * scale,
                    octaves=octaves,
                    repeatx=width,
                    repeaty=height,
                    base=base,
                )
                pattern[y, x] = noise_value
        
        # Map from [-1, 1] to [0, 255]
        pattern = ((pattern + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return pattern
    else:
        # Generate separate Perlin noise for H, S, V channels
        h_noise = np.zeros((height, width), dtype=np.float32)
        s_noise = np.zeros((height, width), dtype=np.float32)
        v_noise = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Generate noise with different base offsets for each channel
                # This ensures H, S, V are independent but reproducible
                h_noise[y, x] = noise.pnoise2(
                    x * h_scale,
                    y * h_scale,
                    octaves=octaves,
                    repeatx=width,
                    repeaty=height,
                    base=base,
                )
                s_noise[y, x] = noise.pnoise2(
                    x * s_scale,
                    y * s_scale,
                    octaves=octaves,
                    repeatx=width,
                    repeaty=height,
                    base=base + 1000,  # Offset to make it independent
                )
                v_noise[y, x] = noise.pnoise2(
                    x * v_scale,
                    y * v_scale,
                    octaves=octaves,
                    repeatx=width,
                    repeaty=height,
                    base=base + 2000,  # Offset to make it independent
                )
        
        # Map noise values to HSV ranges
        # For hue, use percentile-based mapping to ensure uniform color distribution
        # (Perlin noise has a bell-curve distribution, so direct mapping biases toward middle hues)
        # For saturation and value, direct mapping is fine as we want smooth gradients
        
        # Hue: use sorted indices to create uniform distribution
        h_flat = h_noise.flatten()
        h_sorted_indices = np.argsort(h_flat)
        h_uniform = np.zeros_like(h_flat)
        h_uniform[h_sorted_indices] = np.linspace(0.0, 1.0, len(h_flat))
        h = h_uniform.reshape(h_noise.shape) * (hue_range[1] - hue_range[0]) + hue_range[0]
        
        # Saturation and Value: normalize to use full range, then map
        s_min, s_max = s_noise.min(), s_noise.max()
        v_min, v_max = v_noise.min(), v_noise.max()
        
        if s_max > s_min:
            s_normalized = (s_noise - s_min) / (s_max - s_min)  # [0, 1]
        else:
            s_normalized = np.full_like(s_noise, 0.5)
        
        if v_max > v_min:
            v_normalized = (v_noise - v_min) / (v_max - v_min)  # [0, 1]
        else:
            v_normalized = np.full_like(v_noise, 0.5)
        
        s = s_normalized * (saturation_range[1] - saturation_range[0]) + saturation_range[0]
        v = v_normalized * (value_range[1] - value_range[0]) + value_range[0]
        
        # Convert HSV to RGB
        rgb = hsv_to_rgb(h, s, v)
        return rgb


# Pattern generator registry
PATTERN_GENERATORS = {
    "static": generate_static_pattern,
    "perlin": generate_perlin_pattern,
}


def generate_pattern(
    pattern_type: str,
    width: int,
    height: int,
    seed: int | None = None,
    perlin_scale: float | None = None,
    perlin_octaves: int | None = None,
    color: bool | None = None,
    hue_range: tuple[float, float] | None = None,
    saturation_range: tuple[float, float] | None = None,
    value_range: tuple[float, float] | None = None,
    hue_scale: float | None = None,
    saturation_scale: float | None = None,
    value_scale: float | None = None,
) -> np.ndarray:
    """
    Generate a pattern of the specified type.
    
    Args:
        pattern_type: Type of pattern to generate
        width: Pattern width in pixels
        height: Pattern height in pixels
        seed: Random seed for reproducibility
        perlin_scale: Perlin noise scale factor
        perlin_octaves: Number of Perlin noise octaves
        color: If True, generate RGB color pattern; if False, generate grayscale.
               Defaults to True for Perlin patterns, False for others (Perlin only)
        hue_range: (min, max) hue range for color patterns
        saturation_range: (min, max) saturation range for color patterns
        value_range: (min, max) value range for color patterns
        hue_scale: Separate scale for hue noise
        saturation_scale: Separate scale for saturation noise
        value_scale: Separate scale for value noise
    
    Returns:
        Pattern array (grayscale or RGB) as uint8
    """
    if pattern_type not in PATTERN_GENERATORS:
        raise ValueError(
            f"Unknown pattern type: {pattern_type}. "
            f"Supported types: {', '.join(PATTERN_GENERATORS.keys())}"
        )

    generator = PATTERN_GENERATORS[pattern_type]

    if pattern_type == "perlin":
        scale = perlin_scale if perlin_scale is not None else DEFAULT_PERLIN_SCALE
        octaves = perlin_octaves if perlin_octaves is not None else DEFAULT_PERLIN_OCTAVES
        h_range = hue_range if hue_range is not None else DEFAULT_HUE_RANGE
        s_range = saturation_range if saturation_range is not None else DEFAULT_SATURATION_RANGE
        v_range = value_range if value_range is not None else DEFAULT_VALUE_RANGE
        # Default to color=True for Perlin patterns
        use_color = color if color is not None else True
        return generator(
            width,
            height,
            seed=seed,
            scale=scale,
            octaves=octaves,
            color=use_color,
            hue_range=h_range,
            saturation_range=s_range,
            value_range=v_range,
            hue_scale=hue_scale,
            saturation_scale=saturation_scale,
            value_scale=value_scale,
        )
    else:
        if color:
            raise ValueError("Color patterns are only supported for 'perlin' pattern type.")
        return generator(width, height, seed=seed)


# Backward compatibility alias
def generate_noise_pattern(width: int, height: int, seed: int | None = None) -> np.ndarray:
    """Generate a random grayscale noise pattern (backward compatibility)."""
    return generate_static_pattern(width, height, seed=seed)


def calculate_noise_width(image_width: int, custom_width: int | None = None) -> int:
    """Calculate the noise pattern width for an image."""
    if image_width <= 0:
        raise ValueError("Image width must be positive.")
    if custom_width is not None:
        if custom_width <= 0:
            raise ValueError("Custom noise width must be positive.")
        return max(1, min(image_width, int(custom_width)))

    auto_width = max(NOISE_WIDTH_MIN, min(NOISE_WIDTH_MAX, image_width // NOISE_WIDTH_DIVISOR))
    return max(1, min(image_width, auto_width))


def calculate_shift_range(
    image_width: int,
    noise_width: int,
    custom_range: int | None = None,
) -> int:
    """Calculate the shift range used for depth mapping."""
    if image_width <= 0:
        raise ValueError("Image width must be positive.")
    if noise_width <= 0:
        raise ValueError("Noise width must be positive.")

    if custom_range is not None:
        if custom_range <= 0:
            raise ValueError("Custom shift range must be positive.")
        shift = int(custom_range)
    else:
        shift = max(SHIFT_RANGE_MIN, min(SHIFT_RANGE_MAX, image_width // SHIFT_RANGE_DIVISOR))

    max_shift = max(0, noise_width - 1)
    return min(shift, max_shift)


def generate_autostereogram(
    depth_map: np.ndarray,
    noise_width: int | None = None,
    shift_range: int | None = None,
    seed: int | None = None,
    pattern_type: str = DEFAULT_PATTERN_TYPE,
    perlin_scale: float | None = None,
    perlin_octaves: int | None = None,
    color: bool | None = None,
    hue_range: tuple[float, float] | None = None,
    saturation_range: tuple[float, float] | None = None,
    value_range: tuple[float, float] | None = None,
    hue_scale: float | None = None,
    saturation_scale: float | None = None,
    value_scale: float | None = None,
) -> np.ndarray:
    """
    Generate a noise-based autostereogram from a normalized depth map.
    
    Args:
        depth_map: 2D array with normalized depth values [0, 1]
        noise_width: Width of the noise pattern
        shift_range: Maximum pixel shift for depth mapping
        seed: Random seed for pattern generation
        pattern_type: Type of pattern to use
        perlin_scale: Perlin noise scale factor
        perlin_octaves: Number of Perlin noise octaves
        color: If True, generate RGB color stereogram
        hue_range: (min, max) hue range for color patterns
        saturation_range: (min, max) saturation range for color patterns
        value_range: (min, max) value range for color patterns
        hue_scale: Separate scale for hue noise
        saturation_scale: Separate scale for saturation noise
        value_scale: Separate scale for value noise
    
    Returns:
        Stereogram array (grayscale or RGB) as uint8
    """
    if depth_map.ndim != 2:
        raise ValueError("Depth map must be a 2D array.")

    height, width = depth_map.shape
    noise_width = calculate_noise_width(width, noise_width)
    shift_range = calculate_shift_range(width, noise_width, shift_range)

    depth_map = np.clip(depth_map, 0.0, 1.0)
    pattern = generate_pattern(
        pattern_type,
        noise_width,
        height,
        seed=seed,
        perlin_scale=perlin_scale,
        perlin_octaves=perlin_octaves,
        color=color,
        hue_range=hue_range,
        saturation_range=saturation_range,
        value_range=value_range,
        hue_scale=hue_scale,
        saturation_scale=saturation_scale,
        value_scale=value_scale,
    )

    # Determine if pattern is color (3D) or grayscale (2D)
    is_color = pattern.ndim == 3
    
    if is_color:
        stereogram = np.zeros((height, width, 3), dtype=np.uint8)
        stereogram[:, :noise_width, :] = pattern
        
        for row in range(height):
            for col in range(noise_width, width):
                shift = round(depth_map[row, col] * shift_range)
                src_col = col - noise_width + shift
                # Ensure src_col is within valid bounds [0, width-1]
                src_col = max(0, min(width - 1, src_col))
                stereogram[row, col, :] = stereogram[row, src_col, :]
    else:
        stereogram = np.zeros((height, width), dtype=np.uint8)
        stereogram[:, :noise_width] = pattern
        
        for row in range(height):
            for col in range(noise_width, width):
                shift = round(depth_map[row, col] * shift_range)
                src_col = col - noise_width + shift
                # Ensure src_col is within valid bounds [0, width-1]
                src_col = max(0, min(width - 1, src_col))
                stereogram[row, col] = stereogram[row, src_col]

    return stereogram
