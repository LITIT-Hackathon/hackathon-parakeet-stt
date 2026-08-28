"""Command line interface: parakeet transcribe audio.wav -m model.gguf

Transcript goes to stdout, metrics to stderr, so

    parakeet transcribe a.wav -m m.gguf > out.txt

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
    t.add_argument("-m", "--model", default=os.environ.get(ENV_MODEL),
                   help=f"path to the .gguf model (or set ${ENV_MODEL})")
    t.add_argument("--json", action="store_true", dest="as_json",
                   help="emit the full result as JSON on stdout")
    t.add_argument("-q", "--quiet", action="store_true",
                   help="transcript only, no metrics on stderr")
    t.set_defaults(func=_cmd_transcribe)

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


def _cmd_transcribe(args: argparse.Namespace) -> int:
    if not args.model:
        print(f"error: no model given. Pass -m/--model or set ${ENV_MODEL}.",
              file=sys.stderr)
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
