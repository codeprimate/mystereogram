import pytest

from stereogram_generator import depth_estimator


class DummyModel:
    def __init__(self):
        self.moved_to = None
        self.eval_called = False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_load_depth_model_uses_default_device(monkeypatch):
    dummy_model = DummyModel()

    monkeypatch.setattr(depth_estimator, "get_device", lambda: "cpu")
    # Patch transformers module since AutoImageProcessor is imported inside the function
    import transformers
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda _path, **kwargs: "processor",
    )
    monkeypatch.setattr(
        transformers.AutoModelForDepthEstimation,
        "from_pretrained",
        lambda _path, **kwargs: dummy_model,
    )

    model, processor = depth_estimator.load_depth_model()

    assert processor == "processor"
    assert model is dummy_model
    assert dummy_model.moved_to == "cpu"
    assert dummy_model.eval_called is True


def test_load_depth_model_honors_override(monkeypatch):
    dummy_model = DummyModel()

    # Patch transformers module since AutoImageProcessor is imported inside the function
    import transformers
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda _path, **kwargs: "processor",
    )
    monkeypatch.setattr(
        transformers.AutoModelForDepthEstimation,
        "from_pretrained",
        lambda _path, **kwargs: dummy_model,
    )

    model, _processor = depth_estimator.load_depth_model(device="cuda")

    assert model is dummy_model
    assert dummy_model.moved_to == "cuda"


def test_load_depth_model_raises_on_oserror(monkeypatch):
    def raise_error(_path, **kwargs):
        raise OSError("download failed")

    # Patch transformers module since AutoImageProcessor is imported inside the function
    import transformers
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda _path, **kwargs: "processor",
    )
    monkeypatch.setattr(
        transformers.AutoModelForDepthEstimation,
        "from_pretrained",
        raise_error,
    )

    with pytest.raises(OSError, match="Failed to load depth model"):
        depth_estimator.load_depth_model(model_path="bad-model")
