"""Coverage for the package's public surface: version, exports, and that the
two backend flags agree."""

from __future__ import annotations

import parakeet_stt


def test_version_is_a_nonempty_string():
    assert isinstance(parakeet_stt.__version__, str)
    assert parakeet_stt.__version__


def test_all_exports_are_importable():
    for name in parakeet_stt.__all__:
        assert hasattr(parakeet_stt, name), f"__all__ names {name} but it is missing"


def test_backend_flags_are_consistent():
    native = parakeet_stt.is_native()
    assert isinstance(native, bool)
    assert parakeet_stt.backend_name() == ("parakeet.cpp" if native else "stub")
