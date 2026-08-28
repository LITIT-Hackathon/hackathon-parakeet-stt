"""Repeatable smoke test.

Runs green on the stub backend with no model at all, so it is meaningful from
the first commit. When PARAKEET_MODEL points at a real .gguf and the extension
was built native, the same tests assert on real inference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import parakeet_stt
from parakeet_stt import Model, read_wav_mono
from parakeet_stt.audio import AudioError

FIXTURE = Path(__file__).parent / "fixtures" / "tone_16k.wav"
MODEL = os.environ.get("PARAKEET_MODEL")

needs_model = pytest.mark.skipif(
    not MODEL, reason="set PARAKEET_MODEL to run against a real checkpoint"
)


# -- audio layer: no model or extension needed -------------------------------

def test_reads_fixture_as_16k_mono():
    samples, rate = read_wav_mono(FIXTURE)
    assert rate == 16_000
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert np.abs(samples).max() <= 1.0


def test_resamples_and_downmixes(tmp_path):
    # 44.1 kHz stereo in, 16 kHz mono out. This is the silent-failure path:
    # if it ever regresses, transcripts degrade without raising.
    path = tmp_path / "stereo_44k.wav"
    n = 44_100
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44_100)
        w.writeframes((np.zeros(n * 2, dtype=np.int16)).tobytes())

    samples, rate = read_wav_mono(path)
    assert rate == 16_000
    assert samples.ndim == 1
    assert abs(len(samples) - 16_000) < 50  # ~1s of audio, allowing rounding


def test_missing_file_raises_clearly():
    with pytest.raises(AudioError, match="no such file"):
        read_wav_mono("does_not_exist.wav")


# -- extension: builds and imports -------------------------------------------

def test_extension_imports():
    assert parakeet_stt.backend_name() in {"stub", "parakeet.cpp"}
    assert isinstance(parakeet_stt.is_native(), bool)


# -- end to end --------------------------------------------------------------

@needs_model
def test_transcribe_returns_text_and_metrics():
    with Model(MODEL) as m:
        r = m.transcribe(FIXTURE)

    assert r.text.strip(), "transcript was empty"
    assert r.audio_s > 0
    assert r.latency_ms > 0
    assert r.rtf > 0
    assert r.backend == parakeet_stt.backend_name()
    assert set(r.to_dict()) >= {"text", "audio_s", "latency_ms", "rtf"}


@needs_model
def test_cli_json_output():
    out = subprocess.run(
        [sys.executable, "-m", "parakeet_stt.cli",
         "transcribe", str(FIXTURE), "-m", MODEL, "--json"],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(out.stdout)
    assert "text" in payload and "rtf" in payload


@needs_model
@pytest.mark.skipif(not parakeet_stt.is_native(), reason="stub returns canned text")
def test_native_transcript_is_plausible():
    """Guards the 16 kHz trap: a rate mismatch still returns text, but garbage.
    Set PARAKEET_EXPECT to a word you know is in the fixture."""
    expect = os.environ.get("PARAKEET_EXPECT")
    if not expect:
        pytest.skip("set PARAKEET_EXPECT to assert on transcript content")
    with Model(MODEL) as m:
        r = m.transcribe(FIXTURE)
    assert expect.lower() in r.text.lower()
