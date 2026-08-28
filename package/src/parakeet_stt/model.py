"""The public API: Model.transcribe(wav) -> Result.

This is the contract both tracks build against. Track A supplies the native
backend behind it; nothing above this file changes when the stub is replaced.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .audio import TARGET_SAMPLE_RATE, duration_seconds, read_wav_mono


@dataclass(frozen=True)
class Result:
    """A transcript plus the runtime metrics the brief asks us to report."""

    text: str
    audio_s: float          # length of the input audio
    latency_ms: float       # wall clock for the transcription call alone
    rtf: float              # latency / audio duration; < 1.0 is faster than realtime
    model: str              # model filename
    backend: str            # "parakeet.cpp" or "stub"
    load_ms: float = 0.0    # one-off model load, kept separate from inference

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Model:
    """A loaded Parakeet model.

    The model is loaded once on construction and reused across calls, because
    load is expensive and inference is not. Use as a context manager, or call
    close(), to release the native handle deterministically.
    """

    model_path: str | Path
    _backend: _core.Backend = field(init=False, repr=False)
    load_ms: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        t0 = time.perf_counter()
        self._backend = _core.Backend(str(path))
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    # -- public API ---------------------------------------------------------

    def transcribe(self, wav_path: str | Path) -> Result:
        """Transcribe a WAV file. Any sample rate or channel count is accepted;
        it is normalised to 16 kHz mono before inference."""
        samples, sample_rate = read_wav_mono(wav_path)
        audio_s = duration_seconds(samples, sample_rate)

        t0 = time.perf_counter()
        text = self._backend.transcribe_pcm(samples.tolist(), sample_rate)
        latency_s = time.perf_counter() - t0

        return self._result(text, audio_s, latency_s, Path(wav_path).name)

    def transcribe_pcm(self, samples, sample_rate: int = TARGET_SAMPLE_RATE) -> Result:
        """Transcribe raw mono float32 PCM in [-1, 1]."""
        arr = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"expected {TARGET_SAMPLE_RATE} Hz, got {sample_rate}. "
                "Use read_wav_mono() to resample first."
            )
        audio_s = duration_seconds(arr, sample_rate)

        t0 = time.perf_counter()
        text = self._backend.transcribe_pcm(arr.tolist(), sample_rate)
        latency_s = time.perf_counter() - t0

        return self._result(text, audio_s, latency_s, "<pcm>")

    def close(self) -> None:
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend.close()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _result(self, text: str, audio_s: float, latency_s: float, label: str) -> Result:
        return Result(
            text=text.strip(),
            audio_s=round(audio_s, 3),
            latency_ms=round(latency_s * 1000.0, 1),
            rtf=round(latency_s / audio_s, 4) if audio_s > 0 else 0.0,
            model=Path(self.model_path).name,
            backend=_core.backend_name(),
            load_ms=round(self.load_ms, 1),
        )
