"""Shared fixtures and markers for the parakeet-stt test suite.

Most tests here are backend-agnostic: they exercise the Python layer (audio
normalisation, the Model/Result contract, the CLI) which behaves identically on
the stub and the native engine. Tests that genuinely need real inference are
marked `native_only` (and usually `needs_model`); tests that rely on a backend
which ignores the model file's contents are marked `stub_only`.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

import parakeet_stt

FIXTURES = Path(__file__).parent / "fixtures"
TONE = FIXTURES / "tone_16k.wav"
SPEECH_EN = FIXTURES / "speech_en.wav"
SPEECH_DE = FIXTURES / "speech_de.wav"

MODEL = os.environ.get("PARAKEET_MODEL")
NATIVE = parakeet_stt.is_native()

RESULT_FIELDS = {"text", "audio_s", "latency_ms", "rtf", "model", "backend", "load_ms"}

needs_model = pytest.mark.skipif(
    not MODEL, reason="set PARAKEET_MODEL to run the native tier"
)
native_only = pytest.mark.skipif(not NATIVE, reason="native backend only")
stub_only = pytest.mark.skipif(
    NATIVE, reason="stub backend only (needs a backend that ignores model contents)"
)


@pytest.fixture
def stub_or_real_model(tmp_path):
    """A model path valid for whichever backend is compiled in.

    Native: the real PARAKEET_MODEL (skips if unset). Stub: a fabricated file the
    stub accepts. Tests that only exercise the Python layer above the backend
    work with either.
    """
    if NATIVE:
        if not MODEL:
            pytest.skip("set PARAKEET_MODEL to run the native tier")
        return MODEL
    p = tmp_path / "stub-model.gguf"
    p.write_bytes(b"stub")
    return str(p)


@pytest.fixture
def model_and_audio(stub_or_real_model):
    """(model_path, audio_path) that yields a non-empty transcript on either
    backend: real speech under native, the tone under the stub (which returns
    canned text regardless of the audio)."""
    return stub_or_real_model, (SPEECH_EN if NATIVE else TONE)


@pytest.fixture
def write_wav(tmp_path):
    """Factory: write a WAV from a numpy sample array and return its Path.

    `samples` must already be the right integer dtype for `sampwidth`
    (uint8/int16/int32); for stereo pass interleaved samples and n_channels=2.
    """
    def _write(name, samples, sample_rate=16_000, sampwidth=2, n_channels=1):
        path = tmp_path / name
        with wave.open(str(path), "wb") as w:
            w.setnchannels(n_channels)
            w.setsampwidth(sampwidth)
            w.setframerate(sample_rate)
            w.writeframes(np.asarray(samples).tobytes())
        return path

    return _write
