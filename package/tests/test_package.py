"""Coverage for the package's public surface: version, exports, and that the
two backend flags agree."""

from __future__ import annotations

from importlib.metadata import version as _dist_version

import parakeet_stt


def test_version_is_a_nonempty_string():
    assert isinstance(parakeet_stt.__version__, str)
    assert parakeet_stt.__version__


def test_version_matches_package_metadata():
    # __init__.__version__ and pyproject's version are two hand-maintained
    # sources; guard against them drifting apart.
    assert parakeet_stt.__version__ == _dist_version("parakeet-stt")


def test_all_exports_are_importable():
    for name in parakeet_stt.__all__:
        assert hasattr(parakeet_stt, name), f"__all__ names {name} but it is missing"


def test_backend_flags_are_consistent():
    native = parakeet_stt.is_native()
    assert isinstance(native, bool)
    assert parakeet_stt.backend_name() == ("parakeet.cpp" if native else "stub")
