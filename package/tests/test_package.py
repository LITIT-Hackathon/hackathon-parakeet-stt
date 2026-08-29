"""Coverage for the package's public surface: version, exports, and that the
two backend flags agree."""

from __future__ import annotations

import re

import parakeet_stt

# Loose PEP 440: N(.N)*  with optional pre/post/dev/local suffixes.
_PEP440 = re.compile(r"^\d+(\.\d+)*([abc]|rc|\.post|\.dev)?\d*(\+[a-z0-9.]+)?$", re.I)


def test_version_is_a_resolved_pep440_string():
    v = parakeet_stt.__version__
    assert isinstance(v, str) and v
    # __version__ is read from installed metadata; the source-tree fallback
    # should never be what an installed package reports.
    assert v != "0.0.0+unknown", "package metadata was not found at import"
    assert _PEP440.match(v), v


def test_all_exports_are_importable():
    for name in parakeet_stt.__all__:
        assert hasattr(parakeet_stt, name), f"__all__ names {name} but it is missing"


def test_backend_flags_are_consistent():
    native = parakeet_stt.is_native()
    assert isinstance(native, bool)
    assert parakeet_stt.backend_name() == ("parakeet.cpp" if native else "stub")
