import numpy as np
import pytest

from stereogram_generator.stereogram import (
    DEFAULT_PATTERN_TYPE,
    DEFAULT_PERLIN_OCTAVES,
    DEFAULT_PERLIN_SCALE,
    calculate_noise_width,
    calculate_shift_range,
    generate_autostereogram,
    generate_noise_pattern,
    generate_pattern,
    generate_perlin_pattern,
    generate_static_pattern,
    hsv_to_rgb,
)


def test_generate_noise_pattern():
    pattern = generate_noise_pattern(5, 4, seed=123)
    assert pattern.shape == (4, 5)
    assert pattern.dtype == np.uint8
    assert pattern.min() >= 0
    assert pattern.max() <= 255


def test_calculate_noise_width_bounds():
    assert calculate_noise_width(500) == 64
    assert calculate_noise_width(1000) == 66
    assert calculate_noise_width(2000) == 128


def test_calculate_noise_width_custom():
    assert calculate_noise_width(100, custom_width=10) == 10
    assert calculate_noise_width(8, custom_width=10) == 8


def test_calculate_shift_range_bounds():
    assert calculate_shift_range(1000, noise_width=50) == 33
    assert calculate_shift_range(200, noise_width=10) == 9


def test_calculate_shift_range_custom():
    assert calculate_shift_range(1000, noise_width=10, custom_range=3) == 3
    assert calculate_shift_range(1000, noise_width=5, custom_range=10) == 4


def test_generate_autostereogram_shape_and_seed():
    depth_map = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    # Use static pattern to match generate_noise_pattern behavior
    stereogram = generate_autostereogram(
        depth_map, noise_width=2, shift_range=1, seed=7, pattern_type="static"
    )

    assert stereogram.shape == (4, 5)
    assert stereogram.dtype == np.uint8

    pattern = generate_noise_pattern(2, 4, seed=7)
    assert np.array_equal(stereogram[:, :2], pattern)


def test_generate_autostereogram_invalid_depth():
    with pytest.raises(ValueError):
        generate_autostereogram(np.ones((2, 2, 2), dtype=np.float32))


def test_generate_static_pattern():
    """Test static pattern generation (backward compatibility)."""
    pattern = generate_static_pattern(5, 4, seed=123)
    assert pattern.shape == (4, 5)
    assert pattern.dtype == np.uint8
    assert pattern.min() >= 0
    assert pattern.max() <= 255


def test_generate_perlin_pattern():
    """Test Perlin noise pattern generation (grayscale)."""
    pattern = generate_perlin_pattern(10, 8, seed=42, scale=0.1, octaves=1, color=False)
    assert pattern.shape == (8, 10)
    assert pattern.dtype == np.uint8
    assert pattern.min() >= 0
    assert pattern.max() <= 255


def test_generate_perlin_pattern_defaults():
    """Test Perlin pattern with default parameters (now defaults to color)."""
    pattern = generate_perlin_pattern(10, 8, seed=42)
    assert pattern.shape == (8, 10, 3)  # Color is now default
    assert pattern.dtype == np.uint8


def test_generate_perlin_pattern_invalid_scale():
    """Test Perlin pattern with invalid scale."""
    with pytest.raises(ValueError, match="Perlin scale must be positive"):
        generate_perlin_pattern(10, 8, scale=0)


def test_generate_perlin_pattern_invalid_octaves():
    """Test Perlin pattern with invalid octaves."""
    with pytest.raises(ValueError, match="Perlin octaves must be at least 1"):
        generate_perlin_pattern(10, 8, octaves=0)


def test_generate_pattern_static():
    """Test pattern generation dispatcher with static type."""
    pattern = generate_pattern("static", 5, 4, seed=123)
    assert pattern.shape == (4, 5)
    assert pattern.dtype == np.uint8


def test_generate_pattern_perlin():
    """Test pattern generation dispatcher with perlin type (grayscale)."""
    pattern = generate_pattern("perlin", 10, 8, seed=42, perlin_scale=0.1, perlin_octaves=1, color=False)
    assert pattern.shape == (8, 10)
    assert pattern.dtype == np.uint8


def test_generate_pattern_perlin_defaults():
    """Test pattern generation dispatcher with perlin type using defaults (now defaults to color)."""
    pattern = generate_pattern("perlin", 10, 8, seed=42)
    assert pattern.shape == (8, 10, 3)  # Color is now default
    assert pattern.dtype == np.uint8


def test_generate_pattern_invalid_type():
    """Test pattern generation with invalid type."""
    with pytest.raises(ValueError, match="Unknown pattern type"):
        generate_pattern("invalid", 10, 8)


def test_generate_autostereogram_with_pattern_type():
    """Test autostereogram generation with different pattern types."""
    depth_map = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)

    # Test static pattern (default)
    stereogram_static = generate_autostereogram(
        depth_map, noise_width=2, shift_range=1, seed=7, pattern_type="static"
    )
    assert stereogram_static.shape == (4, 5)
    assert stereogram_static.dtype == np.uint8

    # Test perlin pattern (now defaults to color)
    stereogram_perlin = generate_autostereogram(
        depth_map,
        noise_width=2,
        shift_range=1,
        seed=7,
        pattern_type="perlin",
        perlin_scale=0.1,
        perlin_octaves=1,
    )
    assert stereogram_perlin.shape == (4, 5, 3)  # Color is now default
    assert stereogram_perlin.dtype == np.uint8

    # Patterns should be different
    assert not np.array_equal(stereogram_static, stereogram_perlin)


def test_generate_autostereogram_backward_compatibility():
    """Test that default behavior (perlin pattern) is preserved."""
    depth_map = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    stereogram_default = generate_autostereogram(depth_map, noise_width=2, shift_range=1, seed=7)
    stereogram_explicit = generate_autostereogram(
        depth_map, noise_width=2, shift_range=1, seed=7, pattern_type="perlin"
    )
    # Should produce identical results (default is perlin)
    assert np.array_equal(stereogram_default, stereogram_explicit)


def test_hsv_to_rgb():
    """Test HSV to RGB conversion."""
    # Test pure red (H=0, S=1, V=1)
    h = np.array(0.0)
    s = np.array(1.0)
    v = np.array(1.0)
    rgb = hsv_to_rgb(h, s, v)
    assert rgb.shape == (3,)
    assert rgb[0] == 255  # Red channel
    assert rgb[1] == 0
    assert rgb[2] == 0
    
    # Test pure green (H=1/3, S=1, V=1)
    h = np.array(1.0 / 3.0)
    rgb = hsv_to_rgb(h, s, v)
    assert rgb[0] == 0
    assert rgb[1] == 255  # Green channel
    assert rgb[2] == 0
    
    # Test pure blue (H=2/3, S=1, V=1)
    h = np.array(2.0 / 3.0)
    rgb = hsv_to_rgb(h, s, v)
    assert rgb[0] == 0
    assert rgb[1] == 0
    assert rgb[2] == 255  # Blue channel
    
    # Test grayscale (S=0, V=0.5)
    h = np.array(0.0)  # Hue doesn't matter when S=0
    s = np.array(0.0)
    v = np.array(0.5)
    rgb = hsv_to_rgb(h, s, v)
    assert rgb[0] == 127
    assert rgb[1] == 127
    assert rgb[2] == 127
    
    # Test array inputs
    h = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
    s = np.array([1.0, 1.0, 1.0])
    v = np.array([1.0, 1.0, 1.0])
    rgb = hsv_to_rgb(h, s, v)
    assert rgb.shape == (3, 3)
    assert rgb[0, 0] == 255  # Red
    assert rgb[1, 1] == 255  # Green
    assert rgb[2, 2] == 255  # Blue


def test_generate_perlin_pattern_color():
    """Test color Perlin pattern generation."""
    pattern = generate_perlin_pattern(
        10, 8, seed=42, scale=0.1, octaves=1, color=True
    )
    assert pattern.shape == (8, 10, 3)  # RGB channels
    assert pattern.dtype == np.uint8
    assert pattern.min() >= 0
    assert pattern.max() <= 255


def test_generate_perlin_pattern_color_custom_ranges():
    """Test color Perlin pattern with custom HSV ranges."""
    pattern = generate_perlin_pattern(
        10,
        8,
        seed=42,
        scale=0.1,
        octaves=1,
        color=True,
        hue_range=(0.0, 0.5),  # Red to cyan
        saturation_range=(0.8, 1.0),
        value_range=(0.7, 1.0),
    )
    assert pattern.shape == (8, 10, 3)
    assert pattern.dtype == np.uint8


def test_generate_perlin_pattern_color_invalid_ranges():
    """Test color Perlin pattern with invalid HSV ranges."""
    with pytest.raises(ValueError, match="Hue range must be"):
        generate_perlin_pattern(10, 8, color=True, hue_range=(1.0, 0.0))
    
    with pytest.raises(ValueError, match="Saturation range must be"):
        generate_perlin_pattern(10, 8, color=True, saturation_range=(-0.1, 1.0))
    
    with pytest.raises(ValueError, match="Value range must be"):
        generate_perlin_pattern(10, 8, color=True, value_range=(0.0, 1.5))


def test_generate_perlin_pattern_color_separate_scales():
    """Test color Perlin pattern with separate scales for H, S, V."""
    pattern = generate_perlin_pattern(
        10,
        8,
        seed=42,
        scale=0.1,
        octaves=1,
        color=True,
        hue_scale=0.05,
        saturation_scale=0.15,
        value_scale=0.2,
    )
    assert pattern.shape == (8, 10, 3)
    assert pattern.dtype == np.uint8


def test_generate_pattern_color():
    """Test pattern generation dispatcher with color Perlin."""
    pattern = generate_pattern(
        "perlin", 10, 8, seed=42, perlin_scale=0.1, perlin_octaves=1, color=True
    )
    assert pattern.shape == (8, 10, 3)
    assert pattern.dtype == np.uint8


def test_generate_pattern_color_invalid_type():
    """Test that color is only supported for Perlin patterns."""
    with pytest.raises(ValueError, match="Color patterns are only supported"):
        generate_pattern("static", 10, 8, color=True)


def test_generate_autostereogram_color():
    """Test autostereogram generation with color pattern."""
    depth_map = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    stereogram = generate_autostereogram(
        depth_map,
        noise_width=2,
        shift_range=1,
        seed=7,
        pattern_type="perlin",
        perlin_scale=0.1,
        perlin_octaves=1,
        color=True,
    )
    assert stereogram.shape == (4, 5, 3)  # RGB channels
    assert stereogram.dtype == np.uint8


def test_generate_autostereogram_color_vs_grayscale():
    """Test that color and grayscale patterns produce different results."""
    depth_map = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    
    stereogram_grayscale = generate_autostereogram(
        depth_map,
        noise_width=2,
        shift_range=1,
        seed=7,
        pattern_type="perlin",
        perlin_scale=0.1,
        perlin_octaves=1,
        color=False,
    )
    
    stereogram_color = generate_autostereogram(
        depth_map,
        noise_width=2,
        shift_range=1,
        seed=7,
        pattern_type="perlin",
        perlin_scale=0.1,
        perlin_octaves=1,
        color=True,
    )
    
    assert stereogram_grayscale.shape == (4, 5)
    assert stereogram_color.shape == (4, 5, 3)
    
    # Convert grayscale to RGB for comparison
    grayscale_rgb = np.stack([stereogram_grayscale] * 3, axis=-1)
    # They should be different (color has more variation)
    assert not np.array_equal(grayscale_rgb, stereogram_color)
