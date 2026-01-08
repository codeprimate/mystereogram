import math
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from stereogram_generator.utils import (
    get_device,
    load_image,
    resize_image,
    validate_image_format,
)


def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == "cuda"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == "cpu"


def test_get_device_override():
    assert get_device("cpu") == "cpu"
    assert get_device("cuda") == "cuda"


def test_get_device_override_invalid():
    with pytest.raises(ValueError):
        get_device("tpu")
    with pytest.raises(ValueError):
        get_device("mps")


def test_validate_image_format():
    assert validate_image_format("photo.jpg") is True
    assert validate_image_format("photo.jpeg") is True
    assert validate_image_format("photo.png") is True
    assert validate_image_format("photo.txt") is False


def test_load_image_success():
    image = load_image(Path("tests/fixtures/circle.png"))
    assert image.mode == "RGB"
    assert image.size == (256, 256)


def test_load_image_missing(tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        load_image(missing)


def test_load_image_invalid(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        load_image(bad)


def _ratio(size):
    return size[0] / size[1]


def test_resize_image_downscales():
    image = Image.new("RGB", (3000, 1000))
    resized = resize_image(image)
    assert math.isclose(resized.size[0] * resized.size[1], 1_000_000, rel_tol=0.01)
    assert math.isclose(_ratio(resized.size), _ratio(image.size), rel_tol=0.01)


def test_resize_image_upscales():
    image = Image.new("RGB", (500, 500))
    resized = resize_image(image)
    assert math.isclose(resized.size[0] * resized.size[1], 1_000_000, rel_tol=0.01)
    assert math.isclose(_ratio(resized.size), _ratio(image.size), rel_tol=0.01)


def test_resize_image_always_resizes():
    image = Image.new("RGB", (1600, 1000))
    resized = resize_image(image)
    assert math.isclose(resized.size[0] * resized.size[1], 1_000_000, rel_tol=0.01)
    assert math.isclose(_ratio(resized.size), _ratio(image.size), rel_tol=0.01)
