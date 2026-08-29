"""Local speech-to-text on NVIDIA Parakeet with a native C++ inference core."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from ._core import backend_name, is_native
from .audio import AudioError, read_wav_mono
from .model import Model, Result, Word
from .models import DEFAULT_MODEL, ModelError, download_model, list_models

try:  # single source of truth is the version in pyproject.toml
    __version__ = _pkg_version("parakeet-stt")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "Model",
    "Result",
    "Word",
    "AudioError",
    "ModelError",
    "read_wav_mono",
    "backend_name",
    "is_native",
    "download_model",
    "list_models",
    "DEFAULT_MODEL",
    "__version__",
]
