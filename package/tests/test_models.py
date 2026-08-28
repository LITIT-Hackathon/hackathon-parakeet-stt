"""Coverage for models.py: the registry, the checksum-verified downloader, the
`.sha256` marker fast-path, retry, and resolve()'s name-vs-path split.

No network: a `file://` URL to a tiny fixture stands in for the release asset.
"""

from __future__ import annotations

import hashlib
import http.client
from pathlib import Path

import pytest

from parakeet_stt import models as M
from parakeet_stt.models import ModelError, ModelSpec


@pytest.fixture
def asset(tmp_path):
    """A small stand-in for the release GGUF: (path, sha256, size)."""
    data = b"pretend gguf payload " * 4096
    p = tmp_path / "asset.bin"
    p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture
def registered(monkeypatch, asset, tmp_path):
    """Registry -> the local asset; cache -> a temp dir. Returns the ModelSpec."""
    src, sha, size = asset
    spec = ModelSpec(name="tm", url=src.as_uri(), sha256=sha, size=size)
    monkeypatch.setattr(M, "REGISTRY", {"tm": spec})
    monkeypatch.setattr(M, "DEFAULT_MODEL", "tm")
    monkeypatch.setenv("PARAKEET_CACHE_DIR", str(tmp_path / "cache"))
    return spec


# -- registry --------------------------------------------------------------

def test_list_models_reads_the_registry(registered):
    assert M.list_models() == ["tm"]


def test_unknown_name_raises_model_error(registered):
    with pytest.raises(ModelError, match="unknown model"):
        M.download_model("nope")


def test_cache_dir_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PARAKEET_CACHE_DIR", str(tmp_path / "xyz"))
    d = M.cache_dir()
    assert d == tmp_path / "xyz" / "models"
    assert d.is_dir()


# -- resolve() -----------------------------------------------------------------

def test_resolve_passes_a_real_path_through(tmp_path):
    p = tmp_path / "local.gguf"
    p.write_bytes(b"x")
    assert M.resolve(str(p)) == p            # unchanged, no download


def test_resolve_downloads_a_registry_name(registered, asset):
    _, sha, _ = asset
    got = M.resolve("tm")
    assert got.is_file()
    assert hashlib.sha256(got.read_bytes()).hexdigest() == sha


def test_resolve_with_download_false_on_cold_cache_raises(registered):
    with pytest.raises(ModelError, match="not cached"):
        M.resolve("tm", download=False)


# -- download_model() --------------------------------------------------------

def test_download_verifies_and_writes_a_marker(registered, asset):
    _, sha, size = asset
    p = M.download_model("tm")
    assert p.is_file() and p.stat().st_size == size
    marker = p.with_name(p.name + ".sha256")
    assert marker.read_text().strip() == sha


def test_warm_call_does_not_rehash(registered, monkeypatch):
    M.download_model("tm")                                   # populate + marker

    def _boom(*_a, **_k):
        raise AssertionError("warm path must not re-hash the file")

    monkeypatch.setattr(M, "_sha256", _boom)
    assert M.download_model("tm").is_file()                  # marker fast-path


def test_marker_missing_triggers_one_verify(registered, monkeypatch):
    p = M.download_model("tm")
    marker = p.with_name(p.name + ".sha256")
    marker.unlink()                                          # hand-placed-file case

    real = M._sha256
    calls = {"n": 0}

    def counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(M, "_sha256", counting)
    M.download_model("tm")
    assert calls["n"] == 1 and marker.is_file()


def test_size_mismatch_raises_before_the_hash(monkeypatch, asset, tmp_path):
    src, sha, size = asset
    monkeypatch.setenv("PARAKEET_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setattr(M, "REGISTRY",
                        {"bs": ModelSpec("bs", src.as_uri(), sha, size + 1)})
    with pytest.raises(ModelError, match="expected"):
        M.download_model("bs")


def test_checksum_mismatch_raises_and_cleans_up(monkeypatch, asset, tmp_path):
    src, _sha, size = asset
    monkeypatch.setenv("PARAKEET_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setattr(M, "REGISTRY",
                        {"bh": ModelSpec("bh", src.as_uri(), "0" * 64, size)})
    with pytest.raises(ModelError, match="checksum mismatch"):
        M.download_model("bh")
    target = M.cache_dir() / "bh.q8_0.gguf"
    assert not target.exists()
    assert not target.with_name(target.name + ".sha256").exists()


def test_corrupt_cache_is_refetched(registered, asset):
    _, sha, _ = asset
    p = M.download_model("tm")
    p.write_bytes(b"corrupted")
    p.with_name(p.name + ".sha256").write_text(sha)          # stale marker
    p2 = M.download_model("tm")                              # size fails -> refetch
    assert hashlib.sha256(p2.read_bytes()).hexdigest() == sha


# -- retry -----------------------------------------------------------------

def test_download_retries_then_succeeds(registered, monkeypatch):
    real = M._download
    calls = {"n": 0}

    def flaky(url, out, size):
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.client.IncompleteRead(b"", 10)
        return real(url, out, size)

    monkeypatch.setattr(M, "_download", flaky)
    p = M.download_model("tm")
    assert p.is_file() and calls["n"] == 3


def test_download_gives_up_as_model_error(registered, monkeypatch):
    def always_broken(*_a, **_k):
        raise http.client.IncompleteRead(b"", 1)

    monkeypatch.setattr(M, "_download", always_broken)
    with pytest.raises(ModelError, match="after 3 attempts"):
        M.download_model("tm")
