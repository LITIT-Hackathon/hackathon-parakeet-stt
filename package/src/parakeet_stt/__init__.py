"""Local speech-to-text on NVIDIA Parakeet with a native C++ inference core."""

from ._core import backend_name, is_native
from .audio import AudioError, read_wav_mono
from .model import Model, Result

__version__ = "0.1.0"

__all__ = [
    "Model",
    "Result",
    "AudioError",
    "read_wav_mono",
    "backend_name",
    "is_native",
    "__version__",
]
