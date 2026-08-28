"""Coverage for audio.py: the 16 kHz-mono-float32 normalisation and its error
paths. Getting this layer wrong fails silently (confident nonsense), so the
edge cases matter more than the happy path."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from parakeet_stt.audio import (
    AudioError,
    TARGET_SAMPLE_RATE,
    _resample_linear,
    duration_seconds,
    read_wav_mono,
)


# -- format / dtype handling -------------------------------------------------

def test_uint8_is_centered_and_scaled(write_wav):
    # 8-bit WAV is unsigned, centred on 128: 0 -> -1, 128 -> 0, 255 -> ~+1.
    raw = np.array([0, 128, 255, 128], dtype=np.uint8)
    path = write_wav("u8.wav", raw, sampwidth=1)
    samples, rate = read_wav_mono(path)
    assert rate == TARGET_SAMPLE_RATE
    assert samples.dtype == np.float32
    assert samples[0] == pytest.approx(-1.0)
    assert samples[1] == pytest.approx(0.0)
    assert samples[2] == pytest.approx((255 - 128) / 128.0)
    assert np.abs(samples).max() <= 1.0


def test_int32_is_supported(write_wav):
    raw = np.array([0, 2**30, -(2**30)], dtype=np.int32)
    path = write_wav("i32.wav", raw, sampwidth=4)
    samples, rate = read_wav_mono(path)
    assert rate == TARGET_SAMPLE_RATE
    assert samples.dtype == np.float32
    assert np.abs(samples).max() <= 1.0


def test_int16_is_in_unit_range(write_wav):
    raw = np.array([32767, -32768, 0], dtype=np.int16)
    path = write_wav("i16.wav", raw)
    samples, _ = read_wav_mono(path)
    assert samples.min() == pytest.approx(-1.0)
    assert samples.max() <= 1.0


def test_unsupported_sample_width_raises(tmp_path):
    # 24-bit (sampwidth 3) is not in the {1,2,4} map.
    path = tmp_path / "w24.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(16_000)
        w.writeframes(b"\x00\x00\x00" * 1600)
    with pytest.raises(AudioError, match="unsupported sample width"):
        read_wav_mono(path)


def test_corrupt_file_raises_with_conversion_hint(tmp_path):
    path = tmp_path / "notawav.wav"
    path.write_bytes(b"this is plainly not a RIFF/WAVE container")
    with pytest.raises(AudioError, match="not a readable WAV") as exc:
        read_wav_mono(path)
    assert "ffmpeg" in str(exc.value)  # the message tells the user how to fix it


def test_missing_file_raises_clearly():
    with pytest.raises(AudioError, match="no such file"):
        read_wav_mono("does_not_exist.wav")


# -- resampling / downmix ----------------------------------------------------

def test_mono_16k_passes_through_unchanged(write_wav):
    raw = np.arange(-800, 800, dtype=np.int16)
    path = write_wav("m16.wav", raw, sample_rate=16_000)
    samples, rate = read_wav_mono(path)
    assert rate == TARGET_SAMPLE_RATE
    assert len(samples) == len(raw)          # no resampling happened
    assert samples.flags["C_CONTIGUOUS"]


def test_upsamples_8k_to_16k(write_wav):
    path = write_wav("s8k.wav", np.zeros(8_000, np.int16), sample_rate=8_000)
    samples, rate = read_wav_mono(path)
    assert rate == TARGET_SAMPLE_RATE
    assert abs(len(samples) - 16_000) < 50   # ~1 s, doubled


def test_stereo_downmix_averages_channels(write_wav):
    # L = +full scale, R = -full scale -> the mean is silence.
    inter = np.empty(200, dtype=np.int16)
    inter[0::2] = 10_000    # left
    inter[1::2] = -10_000   # right
    path = write_wav("st.wav", inter, sample_rate=16_000, n_channels=2)
    samples, rate = read_wav_mono(path)
    assert rate == TARGET_SAMPLE_RATE
    assert len(samples) == 100
    assert np.abs(samples).max() == pytest.approx(0.0, abs=1e-3)


def test_empty_wav_yields_empty_samples(write_wav):
    path = write_wav("empty.wav", np.zeros(0, np.int16), sample_rate=16_000)
    samples, rate = read_wav_mono(path)
    assert rate == TARGET_SAMPLE_RATE
    assert samples.size == 0


# -- helpers -----------------------------------------------------------------

def test_resample_linear_on_empty_is_empty():
    out = _resample_linear(np.zeros(0, np.float32), 44_100, 16_000)
    assert out.size == 0


def test_resample_linear_length():
    out = _resample_linear(np.zeros(44_100, np.float32), 44_100, 16_000)
    assert abs(len(out) - 16_000) < 2


def test_duration_seconds():
    assert duration_seconds(np.zeros(16_000, np.float32), 16_000) == pytest.approx(1.0)
    assert duration_seconds(np.zeros(0, np.float32), 16_000) == 0.0


def test_resample_linear_preserves_endpoints():
    # A ramp resampled down keeps its first and last value (linear interp), so a
    # bug that shifts or truncates the signal shows up here, not just in length.
    x = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    out = _resample_linear(x, 32_000, 16_000)
    assert len(out) == 500
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[-1] == pytest.approx(1.0, abs=1e-3)
