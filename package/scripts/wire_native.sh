#!/usr/bin/env bash
# Rebuild parakeet-stt against the real parakeet.cpp engine.
#
# This is the Track A -> Track B handoff. Run it once the native library is
# built and a loadable .gguf exists. Everything upstream of the extension
# stays identical; only the compiled backend changes.
#
# Usage:
#   scripts/wire_native.sh /opt/parakeet.cpp /opt/parakeet.cpp/build-shared/libparakeet.so
set -euo pipefail

ROOT="${1:?usage: wire_native.sh <parakeet.cpp root> <libparakeet.so>}"
LIB="${2:?usage: wire_native.sh <parakeet.cpp root> <libparakeet.so>}"

[ -f "$ROOT/include/parakeet_capi.h" ] || { echo "no parakeet_capi.h under $ROOT/include"; exit 1; }
[ -f "$LIB" ] || { echo "library not found: $LIB"; exit 1; }

echo ">> rebuilding parakeet-stt against native backend"
pip install . --force-reinstall --no-deps \
  --config-settings=cmake.define.PARAKEET_ROOT="$ROOT" \
  --config-settings=cmake.define.PARAKEET_LIB="$LIB"

echo ">> verifying"
python -c "import parakeet_stt; assert parakeet_stt.is_native(), 'still on the stub'; print('native backend:', parakeet_stt.backend_name())"

echo ">> done. run: PARAKEET_MODEL=<model.gguf> pytest"
