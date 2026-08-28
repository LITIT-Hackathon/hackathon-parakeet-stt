#!/usr/bin/env bash
# Convert the PROVIDED Parakeet .nemo checkpoint to a GGUF that loads in this
# package's native backend, using parakeet.cpp's own converter at the pinned
# commit (see build_native_lib.sh).
#
# The .gguf committed to this repo was produced by a different converter version
# and does NOT load on the pinned parakeet.cpp; regenerate it from the provided
# .nemo with this script so converter and runtime match.
#
# Usage:
#   scripts/build_model.sh <path/to/parakeet-tdt-0.6b-v3.nemo> [out.gguf] [parakeet.cpp root]
# Defaults: out = ~/parakeet-v3-q8.gguf, root = ~/parakeet.cpp
set -Eeuo pipefail

NEMO="${1:?usage: build_model.sh <parakeet-tdt-0.6b-v3.nemo> [out.gguf] [parakeet.cpp root]}"
OUT="${2:-$HOME/parakeet-v3-q8.gguf}"
ROOT="${3:-$HOME/parakeet.cpp}"

[ -f "$NEMO" ] || { echo "no .nemo at $NEMO -- 'git lfs pull --include=*.nemo' first"; exit 1; }
[ -d "$ROOT" ] || { echo "no parakeet.cpp at $ROOT -- run scripts/build_native_lib.sh first"; exit 1; }

echo ">> setting up converter deps (torch + nemo_toolkit, heavy)"
python3 -m venv "$HOME/cvt"
# shellcheck disable=SC1091
source "$HOME/cvt/bin/activate"
pip install -q --upgrade pip
pip install -q "nemo_toolkit[asr]" gguf

echo ">> converting provided .nemo -> q8_0 GGUF"
python3 "$ROOT/scripts/convert_parakeet_to_gguf.py" \
  --model "$NEMO" --dtype q8_0 --output "$OUT"

echo ">> verifying it loads on the pinned build"
"$ROOT/build-pic/examples/cli/parakeet-cli" info "$OUT" >/dev/null \
  && echo "OK: $OUT loads" \
  || { echo "FAILED to load $OUT"; exit 1; }
