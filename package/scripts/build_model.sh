#!/usr/bin/env bash
# Convert the PROVIDED Parakeet .nemo checkpoint to a GGUF that loads on the
# pinned parakeet.cpp this package builds against.
#
# The .gguf committed to the repo root was produced by a different converter
# version and does NOT load on the pinned build; regenerate from the provided
# .nemo so converter and runtime match.
#
# Self-contained: fetches parakeet.cpp at the pinned commit for its converter,
# and (if parakeet_stt is importable) verifies the result by loading it through
# the installed package. Needs nothing from `pip install` beyond the package.
#
# Usage:
#   scripts/build_model.sh <parakeet-tdt-0.6b-v3.nemo> [out.gguf] [parakeet.cpp dir]
# Defaults: out = ~/parakeet-v3-q8.gguf
#           parakeet.cpp dir = ~/.cache/parakeet-stt/parakeet.cpp
set -Eeuo pipefail

# Keep in sync with PARAKEET_STT_GIT_TAG in package/CMakeLists.txt.
PINNED_SHA="e75de9b6b9b688fd293aa22f7e27aa724ea286f8"

NEMO="${1:?usage: build_model.sh <parakeet-tdt-0.6b-v3.nemo> [out.gguf] [parakeet.cpp dir]}"
OUT="${2:-$HOME/parakeet-v3-q8.gguf}"
ROOT="${3:-$HOME/.cache/parakeet-stt/parakeet.cpp}"

[ -f "$NEMO" ] || { echo "no .nemo at $NEMO -- 'git lfs pull --include=*.nemo' first"; exit 1; }

if [ ! -d "$ROOT/.git" ]; then
  echo ">> fetching parakeet.cpp @ ${PINNED_SHA:0:12} into $ROOT"
  mkdir -p "$(dirname "$ROOT")"
  git clone --quiet https://github.com/mudler/parakeet.cpp "$ROOT"
fi
git -C "$ROOT" fetch --quiet origin "$PINNED_SHA" 2>/dev/null || git -C "$ROOT" fetch --quiet
git -C "$ROOT" checkout --quiet "$PINNED_SHA"
git -C "$ROOT" submodule update --quiet --init --recursive

echo ">> setting up converter deps (torch + nemo_toolkit, heavy, one-off)"
CVT="$HOME/.cache/parakeet-stt/cvt"
python3 -m venv "$CVT"
# shellcheck disable=SC1091
source "$CVT/bin/activate"
pip install -q --upgrade pip
pip install -q "nemo_toolkit[asr]" gguf

echo ">> converting provided .nemo -> q8_0 GGUF"
python3 "$ROOT/scripts/convert_parakeet_to_gguf.py" \
  --model "$NEMO" --dtype q8_0 --output "$OUT"
deactivate

if python3 -c "import parakeet_stt" 2>/dev/null; then
  echo ">> verifying it loads through the installed parakeet_stt"
  python3 -c "from parakeet_stt import Model; Model('$OUT').close(); print('OK: $OUT loads')"
else
  echo ">> parakeet_stt not importable here; skipping load check."
  echo "   after installing the package:  python3 -c \"from parakeet_stt import Model; Model('$OUT').close()\""
fi
