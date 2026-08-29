"""Coverage for cli.py: argument handling, exit codes, and the stdout/stderr
split. The CLI is run as a subprocess so exit codes and stream routing are
asserted for real, the way a shell pipeline sees them."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import parakeet_stt
from conftest import TONE


def _cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "parakeet_stt.cli", *args],
        capture_output=True, text=True, env=env,
    )


# -- no transcription needed -------------------------------------------------

def test_no_subcommand_is_an_error():
    out = _cli()
    assert out.returncode == 2                 # argparse: required subcommand
    assert out.stderr.strip()


def test_version_reports_package_and_backend():
    out = _cli("--version")
    assert out.returncode == 0
    text = out.stdout + out.stderr
    assert "parakeet-stt" in text
    assert parakeet_stt.backend_name() in text


def test_info_reports_backend_and_native():
    out = _cli("info")
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert payload["backend"] == parakeet_stt.backend_name()
    assert payload["native"] == parakeet_stt.is_native()
    assert "version" in payload and "python" in payload


def test_transcribe_requires_the_audio_argument():
    out = _cli("transcribe")                   # missing the required positional
    assert out.returncode == 2                 # argparse usage error
    assert out.stderr.strip()


def test_transcribe_unknown_model_name_returns_2():
    # A name that is neither a registry entry nor a file is reported missing.
    # -m is explicit, so this is independent of $PARAKEET_MODEL.
    out = _cli("transcribe", str(TONE), "-m", "no-such-model")
    assert out.returncode == 2
    assert "not found" in out.stderr.lower()


def test_transcribe_empty_model_returns_2():
    out = _cli("transcribe", str(TONE), "-m", "")
    assert out.returncode == 2
    assert "model" in out.stderr.lower()


def test_download_unknown_model_returns_3():
    out = _cli("download-model", "no-such-model")   # fails before any network
    assert out.returncode == 3
    assert "unknown model" in out.stderr.lower()


def test_list_models_includes_the_default():
    out = _cli("list-models")
    assert out.returncode == 0
    assert parakeet_stt.DEFAULT_MODEL in out.stdout


# -- transcription paths (both tiers via model_and_audio) --------------------

def test_transcribe_missing_audio_returns_2(model_and_audio, tmp_path):
    model, _ = model_and_audio
    out = _cli("transcribe", str(tmp_path / "nope.wav"), "-m", model)
    assert out.returncode == 2
    assert "no such file" in out.stderr.lower()


def test_transcribe_bad_audio_returns_2(model_and_audio, tmp_path):
    model, _ = model_and_audio
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav at all")
    out = _cli("transcribe", str(bad), "-m", model)
    assert out.returncode == 2
    assert "wav" in out.stderr.lower()


def test_transcript_on_stdout_metrics_on_stderr(model_and_audio):
    model, audio = model_and_audio
    out = _cli("transcribe", str(audio), "-m", model)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()                  # transcript on stdout
    assert "RTF" in out.stderr                  # metrics on stderr


def test_quiet_suppresses_metrics(model_and_audio):
    model, audio = model_and_audio
    out = _cli("transcribe", str(audio), "-m", model, "-q")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()
    assert "RTF" not in out.stderr             # -q drops the metrics line


def test_model_read_from_environment(model_and_audio):
    model, audio = model_and_audio
    env = {**os.environ, "PARAKEET_MODEL": model}
    out = _cli("transcribe", str(audio), env=env)   # no -m; picked up from env
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()


def test_json_output_has_the_full_contract(model_and_audio):
    model, audio = model_and_audio
    out = _cli("transcribe", str(audio), "-m", model, "--json")
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert set(payload) == {
        "text", "audio_s", "latency_ms", "rtf", "model", "backend", "words", "load_ms"
    }
    assert payload["text"].strip()
    assert isinstance(payload["words"], list)
