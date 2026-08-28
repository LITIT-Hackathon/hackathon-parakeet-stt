"""Model registry and downloader.

`pip install` does not ship a model. This resolves a short name to a GGUF in a
local cache, fetching it from a release asset on first use and verifying its
SHA-256. An explicit filesystem path is passed through untouched.

A successful verify drops a `<file>.sha256` marker next to the model, so a warm
`Model("<name>")` is a `stat` and a tiny read, not a re-hash of the whole file.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import sys
import tempfile
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
_ATTEMPTS = 3
_RETRIABLE = (urllib.error.URLError, OSError, http.client.HTTPException)


class ModelError(RuntimeError):
    """Raised when a model cannot be resolved, downloaded, or verified."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    size: int  # bytes; a truncated download fails the size check before the hash


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


def _target_for(spec: ModelSpec, dest: Path | str | None) -> Path:
    return Path(dest) if dest else cache_dir() / f"{spec.name}.q8_0.gguf"


def _marker(target: Path) -> Path:
    """Sidecar written after a successful verify; its presence + content means
    `target` was checked against this exact hash and hasn't been touched since."""
    return target.with_name(target.name + ".sha256")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _trusted(target: Path, spec: ModelSpec) -> bool:
    """True if the marker vouches for `target` at `spec`'s hash and the size fits.
    Cheap: no re-hash."""
    m = _marker(target)
    try:
        return (
            target.stat().st_size == spec.size
            and m.read_text().strip() == spec.sha256
        )
    except (OSError, ValueError):
        return False


def _download(url: str, out_path: Path, expected_size: int) -> None:
    """Fetch `url` into `out_path` (a fresh file). No resume; callers retry."""
    req = urllib.request.Request(url)
    # URL is a constant from REGISTRY, not user input.
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        bar = None
        if _tqdm is not None and sys.stderr.isatty():
            bar = _tqdm(total=expected_size, unit="B", unit_scale=True,
                        desc=out_path.name)
        try:
            with out_path.open("wb") as f:
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


def _fetch_verified(spec: ModelSpec, target: Path) -> None:
    """Download `spec` to `target`, size- and SHA-checked, via a process-unique
    temp file so concurrent first-fetches of the same model can't collide."""
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        last: BaseException | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                _download(spec.url, tmp, spec.size)
                break
            except _RETRIABLE as e:
                last = e
                print(f"parakeet-stt: download attempt {attempt} failed ({e})",
                      file=sys.stderr)
        else:
            raise ModelError(
                f"download of {spec.name!r} failed after {_ATTEMPTS} attempts: {last}"
            ) from last

        got_size = tmp.stat().st_size
        if got_size != spec.size:
            raise ModelError(
                f"{spec.name!r} downloaded {got_size} bytes, expected {spec.size}"
            )
        got_sha = _sha256(tmp)
        if got_sha != spec.sha256:
            raise ModelError(
                f"checksum mismatch for {spec.name!r}: "
                f"expected {spec.sha256}, got {got_sha}"
            )
        tmp.replace(target)  # atomic on the same filesystem
        _marker(target).write_text(spec.sha256)
    finally:
        tmp.unlink(missing_ok=True)  # no-op once renamed


def download_model(name: str = DEFAULT_MODEL, *, dest: Path | str | None = None,
                   force: bool = False, allow_download: bool = True) -> Path:
    """Return a local path to model `name`, fetching and verifying if needed.

    `allow_download=False` never touches the network: a missing or unverifiable
    cache entry raises `ModelError` instead.
    """
    try:
        spec = REGISTRY[name]
    except KeyError:
        raise ModelError(
            f"unknown model {name!r}. Known: {', '.join(list_models())}"
        ) from None

    target = _target_for(spec, dest)

    if target.exists() and not force:
        if _trusted(target, spec):
            return target
        # No marker (hand-placed file) or it disagrees: verify once, then trust.
        if _sha256(target) == spec.sha256:
            _marker(target).write_text(spec.sha256)
            return target
        _marker(target).unlink(missing_ok=True)
        if not allow_download:
            raise ModelError(
                f"cached {target.name} fails its checksum and download is disabled"
            )
        print(f"parakeet-stt: cached {target.name} failed its checksum, refetching",
              file=sys.stderr)

    if not allow_download:
        raise ModelError(
            f"model {name!r} is not cached; run `parakeet download-model {name}`"
        )

    print(f"parakeet-stt: fetching model {spec.name!r} "
          f"(~{spec.size / 1e6:.0f} MB) into {target}", file=sys.stderr)
    _fetch_verified(spec, target)
    return target


def resolve(model: str | os.PathLike[str], *, download: bool = True) -> Path:
    """A registry name -> its cached path (downloading on first use unless
    `download=False`). Anything else -> `Path(model)` unchanged, no network."""
    s = os.fspath(model)
    if s in REGISTRY and not Path(s).exists():
        return download_model(s, allow_download=download)
    return Path(s)
