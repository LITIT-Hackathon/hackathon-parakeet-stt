"""Coverage for model.py: the Model lifecycle and the Result contract, above
the backend. These run on either backend via `stub_or_real_model`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from conftest import RESULT_FIELDS, native_only, needs_model, stub_only

from parakeet_stt import Model, Result
from parakeet_stt.audio import TARGET_SAMPLE_RATE


# -- Result dataclass --------------------------------------------------------

def test_result_is_immutable():
    r = Result(text="x", audio_s=1.0, latency_ms=1.0, rtf=0.1, model="m", backend="stub")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.text = "y"  # type: ignore[misc]


def test_result_to_dict_has_exactly_the_contract_fields():
    r = Result(text="hi", audio_s=1.0, latency_ms=2.0, rtf=0.5,
               model="m.gguf", backend="stub", load_ms=3.0)
    d = r.to_dict()
    assert set(d) == RESULT_FIELDS
    assert d["text"] == "hi" and d["model"] == "m.gguf" and d["load_ms"] == 3.0


# -- Model construction ------------------------------------------------------

def test_missing_model_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Model(str(tmp_path / "nope.gguf"))


@native_only
def test_unloadable_gguf_raises(tmp_path):
    # The native engine rejects a file that isn't a valid model; the stub does
    # not look at the bytes, so this is native-only.
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf")
    with pytest.raises(RuntimeError):
        Model(str(bad))


# -- transcribe_pcm contract -------------------------------------------------

def test_transcribe_pcm_rejects_wrong_rate(stub_or_real_model):
    with Model(stub_or_real_model) as m:
        with pytest.raises(ValueError, match="expected 16000"):
            m.transcribe_pcm(np.zeros(1600, np.float32), sample_rate=8_000)


@stub_only
def test_transcribe_pcm_accepts_a_python_list(stub_or_real_model):
    # A plain list (not a numpy array) must work: model.py does np.asarray.
    # Stub-only: exercises the marshalling, not inference (the native path is
    # covered by the smoke tier on real speech).
    with Model(stub_or_real_model) as m:
        r = m.transcribe_pcm([0.0] * 1600, TARGET_SAMPLE_RATE)
    assert isinstance(r, Result)
    assert r.audio_s == pytest.approx(0.1, abs=0.01)


@stub_only
def test_result_reports_the_model_filename(stub_or_real_model):
    # Result.model is the model's basename, regardless of the audio.
    with Model(stub_or_real_model) as m:
        r = m.transcribe_pcm(np.zeros(1600, np.float32))
    assert r.model == Path(stub_or_real_model).name
    assert r.load_ms >= 0.0
    assert r.backend in {"stub", "parakeet.cpp"}


@stub_only
def test_zero_length_audio_gives_rtf_zero(stub_or_real_model):
    # audio_s == 0 must not divide by zero; rtf falls back to 0.0.
    with Model(stub_or_real_model) as m:
        r = m.transcribe_pcm(np.zeros(0, np.float32))
    assert r.audio_s == 0.0
    assert r.rtf == 0.0


# -- lifecycle ---------------------------------------------------------------

def test_close_is_idempotent(stub_or_real_model):
    m = Model(stub_or_real_model)
    m.close()
    m.close()  # second close must be a no-op, not raise


def test_context_manager_closes(stub_or_real_model):
    with Model(stub_or_real_model) as m:
        assert isinstance(m, Model)
    # exiting the context calls close(); calling it again stays safe
    m.close()


@native_only
@needs_model
def test_use_after_close_raises(stub_or_real_model):
    # Only the native backend holds a handle to invalidate; the stub keeps working.
    m = Model(stub_or_real_model)
    m.close()
    with pytest.raises(RuntimeError):
        m.transcribe_pcm(np.zeros(1600, np.float32))
