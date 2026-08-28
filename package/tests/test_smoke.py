"""Track B smoke test.

Two tiers:

  - Stub tier: green with no model and no native build. Exercises the ENTIRE
    Track B surface — audio, API, metrics contract, CLI — because the stub is a
    real compiled backend returning canned text. This is what proves Track B is
    done, independent of Track A.

  - Native tier: gated behind PARAKEET_MODEL. Real inference through
    parakeet.cpp.

The contract, transcribe_pcm and CLI tests run under BOTH tiers via the
`model_audio` fixture, so the Result shape and the CLI are asserted against real
inference too, not only against the stub's canned text.
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

FIXTURES = Path(__file__).parent / "fixtures"
TONE = FIXTURES / "tone_16k.wav"
SPEECH_EN = FIXTURES / "speech_en.wav"
SPEECH_DE = FIXTURES / "speech_de.wav"

MODEL = os.environ.get("PARAKEET_MODEL")
NATIVE = parakeet_stt.is_native()

needs_model = pytest.mark.skipif(
    not MODEL, reason="set PARAKEET_MODEL to run the native tier"
)
native_only = pytest.mark.skipif(not NATIVE, reason="native backend only")

RESULT_FIELDS = {"text", "audio_s", "latency_ms", "rtf", "model", "backend", "load_ms"}


@pytest.fixture
def model_audio(tmp_path):
    """(model_path, audio_path) for whichever backend is compiled in.

    Native: the real model from PARAKEET_MODEL on real speech, so the metrics
    contract and CLI are exercised against actual inference. Stub: a fabricated
    model file the stub accepts, on the tone fixture. The same assertions hold
    for both, which is the point — one contract, two backends.
    """
    if NATIVE:
        if not MODEL:
            pytest.skip("set PARAKEET_MODEL to run the native tier")
        return MODEL, SPEECH_EN
    p = tmp_path / "stub-model.gguf"
    p.write_bytes(b"stub")
    return str(p), TONE


def _cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "parakeet_stt.cli", *args],
        capture_output=True, text=True, env=env,
    )


# ============================ audio layer (no model) ========================

def test_reads_fixture_as_16k_mono():
    samples, rate = read_wav_mono(TONE)
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


# ================================ extension =================================

def test_extension_imports():
    assert parakeet_stt.backend_name() in {"stub", "parakeet.cpp"}
    assert isinstance(parakeet_stt.is_native(), bool)


# ===================== API + metrics contract (both tiers) ==================

def test_transcribe_returns_full_contract(model_audio):
    model, audio = model_audio
    with Model(model) as m:
        r = m.transcribe(audio)

    assert isinstance(r, Result)
    assert r.to_dict().keys() == RESULT_FIELDS
    assert r.text.strip()                       # canned text (stub) or real transcript (native)
    assert isinstance(r.audio_s, float) and r.audio_s > 0
    assert isinstance(r.latency_ms, float) and r.latency_ms >= 0
    assert isinstance(r.rtf, float) and r.rtf >= 0
    assert r.backend == parakeet_stt.backend_name()


def test_transcribe_pcm_path(model_audio):
    model, audio = model_audio
    samples, rate = read_wav_mono(audio)
    with Model(model) as m:
        r = m.transcribe_pcm(samples, rate)
    assert r.text.strip()
    assert r.audio_s > 0


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Model(str(tmp_path / "nope.gguf"))


# ==================================== CLI ===================================

def test_cli_info_runs():
    out = _cli("info")
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert payload["backend"] in {"stub", "parakeet.cpp"}
    assert "native" in payload


def test_cli_transcribe_json(model_audio):
    model, audio = model_audio
    out = _cli("transcribe", str(audio), "-m", model, "--json")
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload.keys() == RESULT_FIELDS
    assert payload["text"].strip()


def test_cli_transcript_to_stdout_metrics_to_stderr(model_audio):
    # The contract that lets `parakeet transcribe a.wav -m m > out.txt` yield a
    # clean transcript file while metrics stay on the terminal.
    model, audio = model_audio
    out = _cli("transcribe", str(audio), "-m", model)
    assert out.returncode == 0
    assert out.stdout.strip()                   # transcript on stdout
    assert "RTF" in out.stderr                   # metrics on stderr


def _clean_env():
    return {k: v for k, v in os.environ.items() if k != "PARAKEET_MODEL"}


def test_cli_unresolvable_model_errors():
    # A name that is neither a registry entry nor an existing file: treated as a
    # path and reported missing.
    out = _cli("transcribe", str(TONE), "-m", "no-such-model", env=_clean_env())
    assert out.returncode == 2
    assert "not found" in out.stderr.lower()


def test_cli_empty_model_errors():
    out = _cli("transcribe", str(TONE), "-m", "", env=_clean_env())
    assert out.returncode == 2
    assert "model" in out.stderr.lower()


def test_cli_download_unknown_model_errors():
    out = _cli("download-model", "no-such-model", env=_clean_env())
    assert out.returncode == 3
    assert "unknown model" in out.stderr.lower()


def test_cli_list_models():
    out = _cli("list-models", env=_clean_env())
    assert out.returncode == 0
    assert "parakeet-tdt-0.6b-v3" in out.stdout


# ===================== native tier: real transcript content =================

@needs_model
@native_only
def test_native_english_is_plausible():
    """Guards the 16 kHz trap: a rate mismatch still returns text, but garbage.
    speech_en.wav is a public-domain reading; 'portrait' is in it."""
    with Model(MODEL) as m:
        r = m.transcribe(SPEECH_EN)
    assert "portrait" in r.text.lower(), r.text


@needs_model
@native_only
def test_native_german_is_plausible():
    """Multilingual v3 on a real DW 'Langsam gesprochene Nachrichten' excerpt
    (see tests/fixtures/NOTICE.md). One model, no per-language config."""
    with Model(MODEL) as m:
        r = m.transcribe(SPEECH_DE)
    assert "sonnenfinsternis" in r.text.lower(), r.text
