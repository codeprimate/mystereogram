import numpy as np
import pytest
import torch
from PIL import Image

from stereogram_generator.depth_estimator import (
    _estimate_depth,
    estimate_depth_map,
    interpolate_depth_map,
    invert_depth_map,
    normalize_depth_map,
    pad_depth_map,
    postprocess_depth_map,
    preprocess_image_for_depth,
)


class DummyProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.ones(1, 3, 2, 2)}


class DummyOutputs:
    def __init__(self, predicted_depth):
        self.predicted_depth = predicted_depth


class DummyModel:
    def __init__(self, predicted_depth):
        self.predicted_depth = predicted_depth
        self.last_input = None

    def __call__(self, pixel_values):
        self.last_input = pixel_values
        return DummyOutputs(self.predicted_depth)


def test_preprocess_image_for_depth_cpu():
    image = Image.new("RGB", (8, 8))
    processor = DummyProcessor()
    pixel_values = preprocess_image_for_depth(image, processor, "cpu")
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.device.type == "cpu"


def test_estimate_depth_runs_model():
    image = Image.new("RGB", (8, 8))
    processor = DummyProcessor()
    expected_depth = torch.ones(1, 2, 2)
    model = DummyModel(expected_depth)

    depth = _estimate_depth(image, model, processor, "cpu")
    assert torch.equal(depth, expected_depth)
    assert model.last_input is not None


def test_interpolate_depth_map_shapes():
    depth = torch.ones(2, 4)
    interpolated = interpolate_depth_map(depth, target_size=(8, 8))
    assert interpolated.shape == (1, 8, 8)


def test_normalize_depth_map_range():
    depth = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    normalized = normalize_depth_map(depth)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)


def test_normalize_depth_map_uniform():
    depth = np.full((2, 2), 5.0, dtype=np.float32)
    normalized = normalize_depth_map(depth)
    assert np.allclose(normalized, 0.0)


def test_postprocess_depth_map():
    depth = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    processed = postprocess_depth_map(depth, original_size=(4, 4))
    assert processed.shape == (4, 4)
    assert np.isclose(processed.min(), 0.0)
    assert np.isclose(processed.max(), 1.0)


def test_invert_depth_map():
    depth = np.array([[0.2, 0.8]], dtype=np.float32)
    inverted = invert_depth_map(depth)
    assert np.allclose(inverted, np.array([[0.8, 0.2]], dtype=np.float32))


def test_pad_depth_map():
    depth = np.array([[0.5, 0.3], [0.7, 0.2]], dtype=np.float32)
    padded = pad_depth_map(depth, padding=50)
    assert padded.shape == (102, 102)  # 2 + 50*2 = 102
    # Check that padding is white (1.0)
    assert np.allclose(padded[:50, :], 1.0)  # Top padding
    assert np.allclose(padded[-50:, :], 1.0)  # Bottom padding
    assert np.allclose(padded[:, :50], 1.0)  # Left padding
    assert np.allclose(padded[:, -50:], 1.0)  # Right padding
    # Check that original content is preserved in center
    assert np.allclose(padded[50:52, 50:52], depth)


def test_pad_depth_map_zero_padding():
    depth = np.array([[0.5, 0.3], [0.7, 0.2]], dtype=np.float32)
    padded = pad_depth_map(depth, padding=0)
    assert padded.shape == depth.shape
    assert np.allclose(padded, depth)


def test_estimate_depth_map_with_dummy_components(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    depth_values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model = DummyModel(depth_values)
    processor = DummyProcessor()

    monkeypatch.setattr(
        "stereogram_generator.depth_estimator.resize_image",
        lambda image: image,
    )

    output_path = tmp_path / "depth.png"
    depth_map = estimate_depth_map(
        image_path,
        device="cpu",
        save_path=output_path,
        model=model,
        processor=processor,
    )

    assert output_path.exists()
    # Depth map should have 50px padding on each side: 8 + 50*2 = 108
    assert depth_map.shape == (108, 108)


def test_estimate_depth_map_with_custom_padding(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    depth_values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model = DummyModel(depth_values)
    processor = DummyProcessor()

    monkeypatch.setattr(
        "stereogram_generator.depth_estimator.resize_image",
        lambda image: image,
    )

    output_path = tmp_path / "depth.png"
    depth_map = estimate_depth_map(
        image_path,
        device="cpu",
        save_path=output_path,
        model=model,
        processor=processor,
        padding=25,
    )

    assert output_path.exists()
    # Depth map should have 25px padding on each side: 8 + 25*2 = 58
    assert depth_map.shape == (58, 58)


