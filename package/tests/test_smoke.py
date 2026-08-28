"""Track B smoke test.

Two tiers:

  - Stub tier: green with no model and no native build. Exercises the ENTIRE
    Track B surface â€” audio, API, metrics contract, CLI â€” because the stub is a
    real compiled backend returning canned text. This is what proves Track B is
    done, independent of Track A.

  - Native tier: gated behind PARAKEET_MODEL. The same shape plus real
    inference, run once Track A's engine is wired in.
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
from parakeet_stt import Model, Result, read_wav_mono
from parakeet_stt.audio import AudioError

FIXTURE = Path(__file__).parent / "fixtures" / "tone_16k.wav"
SPEECH = Path(__file__).parent / "fixtures" / "speech_en.wav"
MODEL = os.environ.get("PARAKEET_MODEL")

needs_model = pytest.mark.skipif(
    not MODEL, reason="set PARAKEET_MODEL to run the native tier"
)

# The stub ignores a model file's contents; the real engine validates it, so a
# fabricated dummy .gguf loads under the stub and is (correctly) refused by
# native. Tests that fabricate a model are therefore stub-only.
stub_only = pytest.mark.skipif(
    parakeet_stt.is_native(),
    reason="stub-only: fabricated model file, native engine rejects it",
)

RESULT_FIELDS = {"text", "audio_s", "latency_ms", "rtf", "model", "backend", "load_ms"}


@pytest.fixture
def stub_model(tmp_path):
    """A model file for the stub tier.

    The stub ignores the file's contents, but Model still checks the path
    exists â€” exactly as the native backend requires â€” so stub and native behave
    identically here. A dummy file lets the full API and CLI run with no real
    checkpoint present.
    """
    p = tmp_path / "stub-model.gguf"
    p.write_bytes(b"stub")
    return str(p)


def _cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "parakeet_stt.cli", *args],
        capture_output=True, text=True, env=env,
    )


# ========================= stub tier: no model needed =======================

# -- audio layer -------------------------------------------------------------

def test_reads_fixture_as_16k_mono():
    samples, rate = read_wav_mono(FIXTURE)
    assert rate == 16_000
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert np.abs(samples).max() <= 1.0


def test_resamples_and_downmixes(tmp_path):
    # 44.1 kHz stereo in, 16 kHz mono out: the silent-failure path. If this
    # regresses, transcripts degrade without any error being raised.
    path = tmp_path / "stereo_44k.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44_100)
        w.writeframes(np.zeros(44_100 * 2, dtype=np.int16).tobytes())

    samples, rate = read_wav_mono(path)
    assert rate == 16_000
    assert samples.ndim == 1
    assert abs(len(samples) - 16_000) < 50  # ~1s, allowing rounding


def test_missing_file_raises_clearly():
    with pytest.raises(AudioError, match="no such file"):
        read_wav_mono("does_not_exist.wav")


# -- extension ---------------------------------------------------------------

def test_extension_imports():
    assert parakeet_stt.backend_name() in {"stub", "parakeet.cpp"}
    assert isinstance(parakeet_stt.is_native(), bool)


# -- API + metrics contract --------------------------------------------------

@stub_only
def test_transcribe_returns_full_contract(stub_model):
    with Model(stub_model) as m:
        r = m.transcribe(FIXTURE)

    assert isinstance(r, Result)
    assert r.to_dict().keys() == RESULT_FIELDS
    assert r.text.strip()                       # stub returns canned text
    assert isinstance(r.audio_s, float) and r.audio_s > 0
    assert isinstance(r.latency_ms, float) and r.latency_ms >= 0
    assert isinstance(r.rtf, float) and r.rtf >= 0
    assert r.backend == parakeet_stt.backend_name()


@stub_only
def test_transcribe_pcm_path(stub_model):
    samples, rate = read_wav_mono(FIXTURE)
    with Model(stub_model) as m:
        r = m.transcribe_pcm(samples, rate)
    assert r.text.strip()
    assert r.audio_s > 0


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Model(str(tmp_path / "nope.gguf"))


# -- CLI ---------------------------------------------------------------------

def test_cli_info_runs():
    out = _cli("info")
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert payload["backend"] in {"stub", "parakeet.cpp"}
    assert "native" in payload


@stub_only
def test_cli_transcribe_json(stub_model):
    out = _cli("transcribe", str(FIXTURE), "-m", stub_model, "--json")
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload.keys() == RESULT_FIELDS
    assert payload["text"].strip()


@stub_only
def test_cli_transcript_to_stdout_metrics_to_stderr(stub_model):
    # The contract that lets `parakeet transcribe a.wav -m m > out.txt` yield a
    # clean transcript file while metrics stay on the terminal.
    out = _cli("transcribe", str(FIXTURE), "-m", stub_model)
    assert out.returncode == 0
    assert out.stdout.strip()                   # transcript on stdout
    assert "RTF" in out.stderr                   # metrics on stderr


def test_cli_missing_model_errors():
    # Run with PARAKEET_MODEL scrubbed from the child env, so "no model" is
    # actually the condition under test even when the suite is run natively
    # with the env var set.
    env = {k: v for k, v in os.environ.items() if k != "PARAKEET_MODEL"}
    out = _cli("transcribe", str(FIXTURE), env=env)   # no -m, no env model
    assert out.returncode == 2
    assert "model" in out.stderr.lower()


# ========================= native tier: needs a model =======================

@needs_model
def test_native_returns_text_and_metrics():
    with Model(MODEL) as m:
        r = m.transcribe(SPEECH)
    assert r.text.strip()
    assert r.rtf > 0
    assert r.backend == parakeet_stt.backend_name()


@needs_model
@pytest.mark.skipif(not parakeet_stt.is_native(), reason="stub returns canned text")
def test_native_transcript_is_plausible():
    """Guards the 16 kHz trap: a rate mismatch still returns text, but garbage.
    PARAKEET_EXPECT overrides the expected word (default: a word from
    speech_en.wav)."""
    expect = os.environ.get("PARAKEET_EXPECT", "portrait")
    with Model(MODEL) as m:
        r = m.transcribe(SPEECH)
    assert expect.lower() in r.text.lower()
