"""Model registry and downloader.

`pip install` does not ship a model. This resolves a short name to a GGUF in a
local cache, fetching it from a release asset on first use and verifying its
SHA-256. An explicit filesystem path is passed through untouched.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import platformdirs

try:  # progress bar is optional
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover
    _tqdm = None

_CHUNK = 1 << 20  # 1 MiB


class ModelError(RuntimeError):
    """Raised when a model cannot be resolved, downloaded, or verified."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    size: int  # bytes; drives the progress bar and a cheap sanity check


# The GGUF the pinned parakeet.cpp loads, produced from the provided .nemo by
# scripts/build_model.sh and published as a release asset. Keep the SHA in sync
# with that asset.
REGISTRY: dict[str, ModelSpec] = {
    "parakeet-tdt-0.6b-v3": ModelSpec(
        name="parakeet-tdt-0.6b-v3",
        url=(
            "https://github.com/LITIT-Hackathon/hackathon-parakeet-stt/"
            "releases/download/models-v1/parakeet-tdt-0.6b-v3.q8_0.gguf"
        ),
        sha256="8ce972f01580135b8801b4b0a462d84dc2d3d5014959adadde0a3e68390d8217",
        size=940_663_712,
    ),
}

DEFAULT_MODEL = "parakeet-tdt-0.6b-v3"


def list_models() -> list[str]:
    return sorted(REGISTRY)


def cache_dir() -> Path:
    """Where downloaded models live. Override with PARAKEET_CACHE_DIR."""
    override = os.environ.get("PARAKEET_CACHE_DIR")
    base = Path(override) if override else Path(platformdirs.user_cache_dir("parakeet-stt"))
    d = base / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_size: int) -> None:
    """Fetch url into dest, resuming a .part file when the server allows it.

    Writes to `<dest>.part` and renames on success, so an interrupted download
    never leaves a half file where a whole one is expected.
    """
    part = dest.parent / (dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")

    # URL is a constant from REGISTRY, not user input.
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        resuming = have > 0 and getattr(resp, "status", 200) == 206
        if have and not resuming:
            have = 0  # server ignored Range; start from scratch

        bar = None
        if _tqdm is not None and sys.stderr.isatty():
            bar = _tqdm(total=expected_size, initial=have, unit="B",
                        unit_scale=True, desc=dest.name)
        try:
            with part.open("ab" if resuming else "wb") as f:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
        finally:
            if bar is not None:
                bar.close()

    part.replace(dest)  # atomic on the same filesystem


def download_model(name: str = DEFAULT_MODEL, *, dest: Path | str | None = None,
                   force: bool = False) -> Path:
    """Return a local path to model `name`, fetching and verifying if needed."""
    try:
        spec = REGISTRY[name]
    except KeyError:
        raise ModelError(
            f"unknown model {name!r}. Known: {', '.join(list_models())}"
        ) from None

    target = Path(dest) if dest else cache_dir() / f"{spec.name}.q8_0.gguf"

    if target.exists() and not force:
        if _sha256(target) == spec.sha256:
            return target
        print(f"parakeet-stt: cached {target.name} failed its checksum, refetching",
              file=sys.stderr)

    print(f"parakeet-stt: fetching model {spec.name!r} "
          f"(~{spec.size / 1e6:.0f} MB) into {target}", file=sys.stderr)
    try:
        _download(spec.url, target, spec.size)
    except (urllib.error.URLError, OSError) as e:
        raise ModelError(f"download of {spec.name!r} failed: {e}") from e

    got = _sha256(target)
    if got != spec.sha256:
        target.unlink(missing_ok=True)
        raise ModelError(
            f"checksum mismatch for {spec.name!r}: "
            f"expected {spec.sha256}, got {got}"
        )
    return target


def resolve(model: str | os.PathLike[str]) -> Path:
    """A registry name -> its cached path (downloading on first use).
    Anything else -> `Path(model)` unchanged."""
    s = os.fspath(model)
    if s in REGISTRY and not Path(s).exists():
        return download_model(s)
    return Path(s)
