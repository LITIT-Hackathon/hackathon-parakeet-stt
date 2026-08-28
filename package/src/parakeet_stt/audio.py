"""WAV loading and normalisation to what Parakeet expects: 16 kHz mono float32.

Getting this wrong does not raise. A 44.1 kHz file decodes to confident
nonsense, and int16 passed where float32 is expected decodes to silence. Both
look like a bad model rather than a bad input, which is why this module
normalises loudly and validates rather than guessing.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000


class AudioError(ValueError):
    """Raised when audio cannot be read or converted."""


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 in [-1, 1], resampled to 16 kHz.

    Returns (samples, sample_rate). sample_rate is always TARGET_SAMPLE_RATE.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"no such file: {path}")

    try:
        with wave.open(str(path), "rb") as w:
            n_channels = w.getnchannels()
            sample_width = w.getsampwidth()
            sample_rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    except wave.Error as e:
        raise AudioError(
            f"{path.name} is not a readable WAV file ({e}). "
            f"Convert it first: ffmpeg -i {path.name} -ar 16000 -ac 1 out.wav"
        ) from e

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        raise AudioError(f"unsupported sample width: {sample_width * 8}-bit")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    # Scale to [-1, 1]. 8-bit WAV is unsigned and centred on 128.
    if dtype is np.uint8:
        samples = (samples - 128.0) / 128.0
    else:
        samples /= float(np.iinfo(dtype).max + 1)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        samples = _resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE)

    return np.ascontiguousarray(samples, dtype=np.float32), TARGET_SAMPLE_RATE


def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear interpolation resample.

    Not as good as a windowed-sinc filter, but speech at 16 kHz is far below
    Nyquist for any common source rate, and the WER difference against a proper
    resampler is not what decides this project. Swap in soxr or librosa if the
    numbers ever suggest otherwise.
    """
    if x.size == 0:
        return x
    duration = x.size / src_rate
    n_out = int(round(duration * dst_rate))
    src_idx = np.linspace(0.0, x.size - 1, num=n_out, dtype=np.float64)
    return np.interp(src_idx, np.arange(x.size, dtype=np.float64), x).astype(np.float32)


def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    return float(samples.size) / float(sample_rate)
