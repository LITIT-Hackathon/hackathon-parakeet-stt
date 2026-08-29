"""The public API: Model.transcribe(wav) -> Result."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .audio import TARGET_SAMPLE_RATE, duration_seconds, read_wav_mono
from .models import DEFAULT_MODEL, resolve


@dataclass(frozen=True)
class Word:
    """One recognised word with its span and confidence."""

    text: str            # the word as transcribed
    start: float          # seconds from the start of the audio
    end: float            # seconds; >= start
    conf: float           # confidence in (0, 1]


@dataclass(frozen=True)
class Result:
    """A transcript plus per-word timing and the runtime metrics."""

    text: str
    audio_s: float                     # length of the input audio
    latency_ms: float                  # wall clock for the transcription call alone
    rtf: float                         # latency / audio duration; < 1.0 is realtime+
    model: str                         # model filename
    backend: str                       # "parakeet.cpp" or "stub"
    words: tuple[Word, ...] = ()        # empty on the stub backend
    load_ms: float = 0.0               # one-off model load, separate from inference

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Model:
    """A loaded Parakeet model.

    The model is loaded once on construction and reused across calls, because
    load is expensive and inference is not. Use as a context manager, or call
    close(), to release the native handle deterministically.
    """

    model_path: str | Path = DEFAULT_MODEL
    download: bool = True
    _backend: _core.Backend = field(init=False, repr=False)
    load_ms: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        # A registry name resolves to a cached file, fetched on first use unless
        # download=False; a filesystem path is taken as-is.
        path = resolve(self.model_path, download=self.download)
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
        doc = self._backend.transcribe_pcm(samples, sample_rate)
        latency_s = time.perf_counter() - t0

        return self._result(doc, audio_s, latency_s)

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
        doc = self._backend.transcribe_pcm(arr, sample_rate)
        latency_s = time.perf_counter() - t0

        return self._result(doc, audio_s, latency_s)

    def close(self) -> None:
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend.close()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _result(self, doc: str, audio_s: float, latency_s: float) -> Result:
        # The backend returns a one-element JSON array (batch entry point,
        # n_clips = 1); the stub returns the same shape with an empty word list.
        clips = json.loads(doc)
        clip = clips[0] if isinstance(clips, list) else clips
        words = tuple(
            Word(text=w["w"], start=w["start"], end=w["end"], conf=w["conf"])
            for w in clip.get("words", ())
        )
        return Result(
            text=str(clip.get("text", "")).strip(),
            audio_s=round(audio_s, 3),
            latency_ms=round(latency_s * 1000.0, 1),
            rtf=round(latency_s / audio_s, 4) if audio_s > 0 else 0.0,
            model=Path(self.model_path).name,
            backend=_core.backend_name(),
            words=words,
            load_ms=round(self.load_ms, 1),
        )
