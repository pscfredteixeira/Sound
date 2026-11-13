"""Device selection runs before torch is imported in the separation child process."""
import pytest
from sampleforge.worker import configure_device


def test_cpu_override_hides_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    configure_device("cpu")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_auto_does_not_modify_cuda_configuration(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    configure_device("auto")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_invalid_device_fails_before_model_loading():
    with pytest.raises(ValueError, match="Dispositivo inválido"):
        configure_device("bogus")
