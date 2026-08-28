"""Command line interface.

    parakeet download-model [name]     fetch a model into the local cache
    parakeet list-models               show known model names
    parakeet transcribe a.wav          transcribe (downloads the default model
                                       on first use)
    parakeet info                      build and backend information

`transcribe` writes the transcript to stdout and the metrics to stderr, so

    parakeet transcribe a.wav > out.txt

gives a clean transcript file while the numbers stay visible on the terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from ._core import backend_name, is_native
from .audio import AudioError
from .model import Model
from .models import DEFAULT_MODEL, ModelError, download_model, list_models

ENV_MODEL = "PARAKEET_MODEL"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parakeet",
        description="Local speech-to-text on NVIDIA Parakeet.",
    )
    p.add_argument("--version", action="version",
                   version=f"parakeet-stt {__version__} (backend: {backend_name()})")

    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe a WAV file")
    t.add_argument("audio", help="path to a WAV file (any rate; resampled to 16 kHz mono)")
    t.add_argument("-m", "--model",
                   default=os.environ.get(ENV_MODEL) or DEFAULT_MODEL,
                   help=f"a known model name or a path to a .gguf "
                        f"(default: ${ENV_MODEL} or {DEFAULT_MODEL!r})")
    t.add_argument("--json", action="store_true", dest="as_json",
                   help="emit the full result as JSON on stdout")
    t.add_argument("-q", "--quiet", action="store_true",
                   help="transcript only, no metrics on stderr")
    t.set_defaults(func=_cmd_transcribe)

    d = sub.add_parser("download-model", help="fetch a model into the local cache")
    d.add_argument("name", nargs="?", default=DEFAULT_MODEL,
                   help=f"model name (default: {DEFAULT_MODEL!r})")
    d.add_argument("--force", action="store_true", help="re-download even if cached")
    d.set_defaults(func=_cmd_download)

    lm = sub.add_parser("list-models", help="show known model names")
    lm.set_defaults(func=_cmd_list_models)

    i = sub.add_parser("info", help="show build and backend information")
    i.set_defaults(func=_cmd_info)

    return p


def _cmd_info(args: argparse.Namespace) -> int:
    print(json.dumps({
        "version": __version__,
        "backend": backend_name(),
        "native": is_native(),
        "python": sys.version.split()[0],
    }, indent=2))
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    for name in list_models():
        print(name)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        path = download_model(args.name, force=args.force)
    except ModelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    print(path)
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    if not args.model:
        print("error: no model. Pass -m NAME|PATH, set $PARAKEET_MODEL, or "
              "run 'parakeet download-model'.", file=sys.stderr)
        return 2

    try:
        with Model(args.model) as model:
            result = model.transcribe(args.audio)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ModelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except RuntimeError as e:
        print(f"error: transcription failed: {e}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(result.text)
    if not args.quiet:
        print(
            f"\n  backend {result.backend} | audio {result.audio_s:.2f}s "
            f"| load {result.load_ms:.0f}ms | infer {result.latency_ms:.0f}ms "
            f"| RTF {result.rtf:.3f}",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
