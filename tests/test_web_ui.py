"""Tests for web_ui module."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from stereogram_generator.web_ui import (
    ProcessingConfig,
    StereogramConfig,
    create_interface,
    get_cached_model,
    main,
    process_image_for_web,
)


class TestStereogramConfig:
    """Tests for StereogramConfig dataclass."""

    def test_stereogram_config_creation(self):
        """Test creating a StereogramConfig with all parameters."""
        config = StereogramConfig(
            pattern_type="perlin",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=3,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.5,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=0.05,
            saturation_scale=0.1,
            value_scale=0.15,
            noise_width=64,
            shift_range=32,
        )
        assert config.pattern_type == "perlin"
        assert config.mono is False
        assert config.perlin_scale == 0.1
        assert config.noise_width == 64
        assert config.shift_range == 32

    def test_stereogram_config_with_none_values(self):
        """Test StereogramConfig with optional None values."""
        config = StereogramConfig(
            pattern_type="static",
            mono=True,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        assert config.hue_scale is None
        assert config.noise_width is None


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_processing_config_creation(self):
        """Test creating a ProcessingConfig with all parameters."""
        config = ProcessingConfig(
            device="cuda",
            padding=50,
            show_depth_map=True,
            input_filename="test.jpg",
        )
        assert config.device == "cuda"
        assert config.padding == 50
        assert config.show_depth_map is True
        assert config.input_filename == "test.jpg"

    def test_processing_config_with_defaults(self):
        """Test ProcessingConfig with default values."""
        config = ProcessingConfig(
            device="auto",
            padding=50,
            show_depth_map=False,
        )
        assert config.input_filename is None


class TestGetCachedModel:
    """Tests for get_cached_model function."""

    def test_get_cached_model_loads_on_first_call(self, monkeypatch):
        """Test that model is loaded on first call."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def mock_load_model(device=None):
            return (mock_model, mock_processor)

        monkeypatch.setattr(
            "stereogram_generator.web_ui.load_depth_model", mock_load_model
        )

        # Clear cache
        import stereogram_generator.web_ui as web_ui_module

        web_ui_module._model_cache.clear()

        result_model, result_processor = get_cached_model("cpu")

        assert result_model is mock_model
        assert result_processor is mock_processor

    def test_get_cached_model_uses_cache_on_second_call(self, monkeypatch):
        """Test that model is cached and reused on second call."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        call_count = {"count": 0}

        def mock_load_model(device=None):
            call_count["count"] += 1
            return (mock_model, mock_processor)

        monkeypatch.setattr(
            "stereogram_generator.web_ui.load_depth_model", mock_load_model
        )

        # Clear cache
        import stereogram_generator.web_ui as web_ui_module

        web_ui_module._model_cache.clear()

        # First call
        get_cached_model("cpu")
        assert call_count["count"] == 1

        # Second call with same device
        get_cached_model("cpu")
        assert call_count["count"] == 1  # Should not call again

    def test_get_cached_model_loads_separate_models_for_different_devices(
        self, monkeypatch
    ):
        """Test that different devices get separate cached models."""
        cpu_model = MagicMock()
        cpu_processor = MagicMock()
        cuda_model = MagicMock()
        cuda_processor = MagicMock()

        def mock_load_model(device=None):
            if device == "cpu":
                return (cpu_model, cpu_processor)
            elif device == "cuda":
                return (cuda_model, cuda_processor)
            return (MagicMock(), MagicMock())

        monkeypatch.setattr(
            "stereogram_generator.web_ui.load_depth_model", mock_load_model
        )

        # Clear cache
        import stereogram_generator.web_ui as web_ui_module

        web_ui_module._model_cache.clear()

        cpu_result = get_cached_model("cpu")
        cuda_result = get_cached_model("cuda")

        assert cpu_result[0] is cpu_model
        assert cuda_result[0] is cuda_model
        assert cpu_result[0] is not cuda_result[0]


class TestProcessImageForWeb:
    """Tests for process_image_for_web function."""

    def test_process_image_for_web_with_none_image(self):
        """Test that None image returns appropriate message."""
        config = StereogramConfig(
            pattern_type="static",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu", padding=50, show_depth_map=False
        )

        result = process_image_for_web(None, config, processing_config)

        assert result[0] is None
        assert result[1] is None
        assert "Please upload an image" in result[2]

    def test_process_image_for_web_happy_path(self, tmp_path, monkeypatch):
        """Test successful image processing."""
        # Create test image
        input_image = Image.new("RGB", (100, 100), color="red")
        input_path = tmp_path / "test.png"
        input_image.save(input_path)

        # Mock dependencies
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def mock_get_cached_model(device):
            return (mock_model, mock_processor)

        def mock_normalize_image(img):
            return img

        def mock_resize_image(img):
            return img

        def mock_estimate_depth(img, model, processor, device):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_postprocess_depth_map(tensor, size):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_pad_depth_map(depth_map, padding):
            h, w = depth_map.shape
            padded = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.float32)
            padded[padding:-padding, padding:-padding] = depth_map
            return padded

        def mock_generate_autostereogram(*args, **kwargs):
            return np.zeros((200, 200), dtype=np.uint8)

        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.normalize_image", mock_normalize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.resize_image", mock_resize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui._estimate_depth", mock_estimate_depth
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.postprocess_depth_map",
            mock_postprocess_depth_map,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.pad_depth_map", mock_pad_depth_map
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.generate_autostereogram",
            mock_generate_autostereogram,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.invert_depth_map",
            lambda x: x,  # No-op
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu", padding=50, show_depth_map=False
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is not None
        assert Path(result[0]).exists()
        assert result[1] is None  # show_depth_map is False
        assert "Processing time" in result[2]
        assert "Output size" in result[2]
        assert "Device" in result[2]

    def test_process_image_for_web_with_depth_map(self, tmp_path, monkeypatch):
        """Test processing with depth map visualization enabled."""
        input_image = Image.new("RGB", (100, 100), color="blue")
        input_path = tmp_path / "test.png"
        input_image.save(input_path)

        # Mock dependencies
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def mock_get_cached_model(device):
            return (mock_model, mock_processor)

        def mock_normalize_image(img):
            return img

        def mock_resize_image(img):
            return img

        def mock_estimate_depth(img, model, processor, device):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_postprocess_depth_map(tensor, size):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_pad_depth_map(depth_map, padding):
            h, w = depth_map.shape
            padded = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.float32)
            padded[padding:-padding, padding:-padding] = depth_map
            return padded

        def mock_generate_autostereogram(*args, **kwargs):
            return np.zeros((200, 200), dtype=np.uint8)

        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.normalize_image", mock_normalize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.resize_image", mock_resize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui._estimate_depth", mock_estimate_depth
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.postprocess_depth_map",
            mock_postprocess_depth_map,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.pad_depth_map", mock_pad_depth_map
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.generate_autostereogram",
            mock_generate_autostereogram,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.invert_depth_map",
            lambda x: x,  # No-op
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu", padding=50, show_depth_map=True
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is not None
        assert result[1] is not None  # Depth map should be created
        assert isinstance(result[1], Image.Image)

    def test_process_image_for_web_with_mono(self, tmp_path, monkeypatch):
        """Test processing with monochrome stereogram."""
        input_image = Image.new("RGB", (100, 100), color="green")
        input_path = tmp_path / "test.png"
        input_image.save(input_path)

        # Mock dependencies
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def mock_get_cached_model(device):
            return (mock_model, mock_processor)

        def mock_normalize_image(img):
            return img

        def mock_resize_image(img):
            return img

        def mock_estimate_depth(img, model, processor, device):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_postprocess_depth_map(tensor, size):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_pad_depth_map(depth_map, padding):
            h, w = depth_map.shape
            padded = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.float32)
            padded[padding:-padding, padding:-padding] = depth_map
            return padded

        def mock_generate_autostereogram(*args, **kwargs):
            return np.zeros((200, 200), dtype=np.uint8)

        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.normalize_image", mock_normalize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.resize_image", mock_resize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui._estimate_depth", mock_estimate_depth
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.postprocess_depth_map",
            mock_postprocess_depth_map,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.pad_depth_map", mock_pad_depth_map
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.generate_autostereogram",
            mock_generate_autostereogram,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.invert_depth_map",
            lambda x: x,  # No-op
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=True,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu", padding=50, show_depth_map=False
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is not None
        assert "Processing time" in result[2]

    def test_process_image_for_web_with_auto_device(self, tmp_path, monkeypatch):
        """Test processing with auto device selection."""
        input_image = Image.new("RGB", (100, 100), color="yellow")
        input_path = tmp_path / "test.png"
        input_image.save(input_path)

        # Mock dependencies
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def mock_get_device():
            return "cuda"

        def mock_get_cached_model(device):
            return (mock_model, mock_processor)

        def mock_normalize_image(img):
            return img

        def mock_resize_image(img):
            return img

        def mock_estimate_depth(img, model, processor, device):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_postprocess_depth_map(tensor, size):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_pad_depth_map(depth_map, padding):
            h, w = depth_map.shape
            padded = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.float32)
            padded[padding:-padding, padding:-padding] = depth_map
            return padded

        def mock_generate_autostereogram(*args, **kwargs):
            return np.zeros((200, 200), dtype=np.uint8)

        monkeypatch.setattr("stereogram_generator.web_ui.get_device", mock_get_device)
        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.normalize_image", mock_normalize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.resize_image", mock_resize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui._estimate_depth", mock_estimate_depth
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.postprocess_depth_map",
            mock_postprocess_depth_map,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.pad_depth_map", mock_pad_depth_map
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.generate_autostereogram",
            mock_generate_autostereogram,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.invert_depth_map",
            lambda x: x,  # No-op
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="auto", padding=50, show_depth_map=False
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is not None
        assert "cuda" in result[2]  # Should show selected device

    def test_process_image_for_web_error_handling(self, tmp_path, monkeypatch):
        """Test error handling in process_image_for_web."""
        input_image = Image.new("RGB", (100, 100), color="red")
        input_path = tmp_path / "test.png"
        input_image.save(input_path)

        def mock_get_cached_model(device):
            raise RuntimeError("Model loading failed")

        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu", padding=50, show_depth_map=False
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is None
        assert result[1] is None
        assert "Error" in result[2]
        assert "Model loading failed" in result[2]

    def test_process_image_for_web_with_input_filename(self, tmp_path, monkeypatch):
        """Test that input filename is used in output filename."""
        input_image = Image.new("RGB", (100, 100), color="purple")
        input_path = tmp_path / "my_image.jpg"
        input_image.save(input_path)

        # Mock dependencies
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def mock_get_cached_model(device):
            return (mock_model, mock_processor)

        def mock_normalize_image(img):
            return img

        def mock_resize_image(img):
            return img

        def mock_estimate_depth(img, model, processor, device):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_postprocess_depth_map(tensor, size):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_pad_depth_map(depth_map, padding):
            h, w = depth_map.shape
            padded = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.float32)
            padded[padding:-padding, padding:-padding] = depth_map
            return padded

        def mock_generate_autostereogram(*args, **kwargs):
            return np.zeros((200, 200), dtype=np.uint8)

        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.normalize_image", mock_normalize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.resize_image", mock_resize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui._estimate_depth", mock_estimate_depth
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.postprocess_depth_map",
            mock_postprocess_depth_map,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.pad_depth_map", mock_pad_depth_map
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.generate_autostereogram",
            mock_generate_autostereogram,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.invert_depth_map",
            lambda x: x,  # No-op
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu",
            padding=50,
            show_depth_map=False,
            input_filename="my_image.jpg",
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is not None
        output_path = Path(result[0])
        assert "my_image" in output_path.name

    def test_process_image_for_web_with_invert_depth(self, tmp_path, monkeypatch):
        """Test processing with INVERT_DEPTH enabled."""
        input_image = Image.new("RGB", (100, 100), color="orange")
        input_path = tmp_path / "test.png"
        input_image.save(input_path)

        # Mock dependencies
        mock_model = MagicMock()
        mock_processor = MagicMock()
        invert_called = {"called": False}

        def mock_get_cached_model(device):
            return (mock_model, mock_processor)

        def mock_normalize_image(img):
            return img

        def mock_resize_image(img):
            return img

        def mock_estimate_depth(img, model, processor, device):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_postprocess_depth_map(tensor, size):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_invert_depth_map(depth_map):
            invert_called["called"] = True
            return depth_map

        def mock_pad_depth_map(depth_map, padding):
            h, w = depth_map.shape
            padded = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.float32)
            padded[padding:-padding, padding:-padding] = depth_map
            return padded

        def mock_generate_autostereogram(*args, **kwargs):
            return np.zeros((200, 200), dtype=np.uint8)

        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.normalize_image", mock_normalize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.resize_image", mock_resize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui._estimate_depth", mock_estimate_depth
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.postprocess_depth_map",
            mock_postprocess_depth_map,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.pad_depth_map", mock_pad_depth_map
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.generate_autostereogram",
            mock_generate_autostereogram,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.invert_depth_map", mock_invert_depth_map
        )
        # Set INVERT_DEPTH to True
        monkeypatch.setattr(
            "stereogram_generator.web_ui.INVERT_DEPTH", True
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=False,
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu", padding=50, show_depth_map=False
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is not None
        assert invert_called["called"] is True

    def test_process_image_for_web_with_grayscale_stereogram(self, tmp_path, monkeypatch):
        """Test processing with grayscale stereogram (2D array)."""
        input_image = Image.new("RGB", (100, 100), color="cyan")
        input_path = tmp_path / "test.png"
        input_image.save(input_path)

        # Mock dependencies
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def mock_get_cached_model(device):
            return (mock_model, mock_processor)

        def mock_normalize_image(img):
            return img

        def mock_resize_image(img):
            return img

        def mock_estimate_depth(img, model, processor, device):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_postprocess_depth_map(tensor, size):
            return np.zeros((100, 100), dtype=np.float32)

        def mock_pad_depth_map(depth_map, padding):
            h, w = depth_map.shape
            padded = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.float32)
            padded[padding:-padding, padding:-padding] = depth_map
            return padded

        # Return 2D array (grayscale) instead of 3D (color)
        def mock_generate_autostereogram(*args, **kwargs):
            return np.zeros((200, 200), dtype=np.uint8)  # 2D array

        monkeypatch.setattr(
            "stereogram_generator.web_ui.get_cached_model", mock_get_cached_model
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.normalize_image", mock_normalize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.resize_image", mock_resize_image
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui._estimate_depth", mock_estimate_depth
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.postprocess_depth_map",
            mock_postprocess_depth_map,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.pad_depth_map", mock_pad_depth_map
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.generate_autostereogram",
            mock_generate_autostereogram,
        )
        monkeypatch.setattr(
            "stereogram_generator.web_ui.invert_depth_map",
            lambda x: x,  # No-op
        )

        config = StereogramConfig(
            pattern_type="static",
            mono=True,  # Grayscale
            perlin_scale=0.1,
            perlin_octaves=1,
            hue_range_min=0.0,
            hue_range_max=1.0,
            saturation_range_min=0.0,
            saturation_range_max=1.0,
            value_range_min=0.0,
            value_range_max=1.0,
            hue_scale=None,
            saturation_scale=None,
            value_scale=None,
            noise_width=None,
            shift_range=None,
        )
        processing_config = ProcessingConfig(
            device="cpu", padding=50, show_depth_map=False
        )

        result = process_image_for_web(str(input_path), config, processing_config)

        assert result[0] is not None
        # Verify the image was saved (grayscale mode "L")
        saved_image = Image.open(result[0])
        assert saved_image.mode == "L"  # Grayscale mode


class TestCreateInterface:
    """Tests for create_interface function."""

    def test_create_interface_returns_gradio_blocks(self):
        """Test that create_interface returns a Gradio Blocks object."""
        interface = create_interface()
        assert interface is not None
        # Verify it's a Gradio Blocks instance
        import gradio as gr

        assert isinstance(interface, gr.Blocks)


class TestMain:
    """Tests for main function."""

    def test_main_with_default_args(self, monkeypatch):
        """Test main function with default arguments."""
        mock_interface = MagicMock()
        mock_interface.launch.return_value = None

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web"]):
            result = main()

        assert result == 0
        mock_interface.launch.assert_called_once()
        call_kwargs = mock_interface.launch.call_args[1]
        assert call_kwargs["server_port"] == 7860
        assert call_kwargs["server_name"] == "127.0.0.1"
        assert call_kwargs["share"] is False
        assert call_kwargs["inbrowser"] is True

    def test_main_with_custom_port(self, monkeypatch):
        """Test main function with custom port."""
        mock_interface = MagicMock()
        mock_interface.launch.return_value = None

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web", "--port", "8080"]):
            result = main()

        assert result == 0
        call_kwargs = mock_interface.launch.call_args[1]
        assert call_kwargs["server_port"] == 8080

    def test_main_with_custom_host(self, monkeypatch):
        """Test main function with custom host."""
        mock_interface = MagicMock()
        mock_interface.launch.return_value = None

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web", "--host", "0.0.0.0"]):
            result = main()

        assert result == 0
        call_kwargs = mock_interface.launch.call_args[1]
        assert call_kwargs["server_name"] == "0.0.0.0"

    def test_main_with_share(self, monkeypatch):
        """Test main function with --share flag."""
        mock_interface = MagicMock()
        mock_interface.launch.return_value = None

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web", "--share"]):
            result = main()

        assert result == 0
        call_kwargs = mock_interface.launch.call_args[1]
        assert call_kwargs["share"] is True

    def test_main_with_no_browser(self, monkeypatch):
        """Test main function with --no-browser flag."""
        mock_interface = MagicMock()
        mock_interface.launch.return_value = None

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web", "--no-browser"]):
            result = main()

        assert result == 0
        call_kwargs = mock_interface.launch.call_args[1]
        assert call_kwargs["inbrowser"] is False

    def test_main_with_invalid_port(self, monkeypatch, capsys):
        """Test main function with invalid port."""
        with patch("sys.argv", ["mystereogram-web", "--port", "70000"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Port must be between 1 and 65535" in captured.out

    def test_main_with_port_already_in_use(self, monkeypatch, capsys):
        """Test main function when port is already in use."""
        mock_interface = MagicMock()
        mock_interface.launch.side_effect = OSError("Address already in use")

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web", "--port", "7860"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Port 7860 is already in use" in captured.out

    def test_main_with_keyboard_interrupt(self, monkeypatch):
        """Test main function handling KeyboardInterrupt."""
        mock_interface = MagicMock()
        mock_interface.launch.side_effect = KeyboardInterrupt()

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web"]):
            result = main()

        assert result == 0

    def test_main_with_generic_error(self, monkeypatch, capsys):
        """Test main function handling generic errors."""
        mock_interface = MagicMock()
        mock_interface.launch.side_effect = Exception("Generic error")

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Error: Generic error" in captured.out

    def test_main_with_log_level(self, monkeypatch):
        """Test main function with custom log level."""
        mock_interface = MagicMock()
        mock_interface.launch.return_value = None

        def mock_create_interface():
            return mock_interface

        monkeypatch.setattr(
            "stereogram_generator.web_ui.create_interface", mock_create_interface
        )

        with patch("sys.argv", ["mystereogram-web", "--log-level", "DEBUG"]):
            result = main()

        assert result == 0
        # Verify logging was configured (can't easily test the level, but no error means it worked)

