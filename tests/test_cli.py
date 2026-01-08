from pathlib import Path

import numpy as np
from PIL import Image

from stereogram_generator import cli


def test_main_happy_path(tmp_path, monkeypatch):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    Image.new("RGB", (8, 8), color="white").save(input_path)

    def fake_estimate_depth_map(*_args, **_kwargs):
        return np.zeros((8, 8), dtype=np.float32)

    def fake_generate_autostereogram(*_args, **_kwargs):
        return np.zeros((8, 8), dtype=np.uint8)

    monkeypatch.setattr(cli, "estimate_depth_map", fake_estimate_depth_map)
    monkeypatch.setattr(cli, "generate_autostereogram", fake_generate_autostereogram)

    exit_code = cli.main(["-i", str(input_path), "-o", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()


def test_main_missing_input(tmp_path):
    output_path = tmp_path / "output.png"
    exit_code = cli.main(["-i", "missing.png", "-o", str(output_path)])
    assert exit_code == 1


def test_main_invalid_noise_width(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (8, 8), color="white").save(input_path)

    exit_code = cli.main(["-i", str(input_path), "-o", str(output_path), "--noise-width", "-1"])
    assert exit_code == 1
