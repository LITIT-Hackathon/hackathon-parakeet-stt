#!/usr/bin/env bash
# Rebuild parakeet-stt against the real parakeet.cpp engine.
#
# This is the Track A -> Track B handoff. Run it once the native library is
# built (see scripts/build_native_lib.sh) and a loadable .gguf exists.
# Everything upstream of the extension stays identical; only the compiled
# backend changes.
#
# Usage:
#   scripts/wire_native.sh <parakeet.cpp root> <libparakeet.so>
set -euo pipefail

ROOT="${1:?usage: wire_native.sh <parakeet.cpp root> <libparakeet.so>}"
LIB="${2:?usage: wire_native.sh <parakeet.cpp root> <libparakeet.so>}"

[ -f "$ROOT/include/parakeet_capi.h" ] || { echo "no parakeet_capi.h under $ROOT/include"; exit 1; }
[ -f "$LIB" ] || { echo "library not found: $LIB"; exit 1; }

# The extension links against LIB and must find it (and the ggml libraries that
# sit beside it) at import time. Bake the library's directory in as an rpath so
# no LD_LIBRARY_PATH is needed at runtime.
LIBDIR="$(cd "$(dirname "$LIB")" && pwd)"

echo ">> rebuilding parakeet-stt against native backend"
pip install . --force-reinstall --no-deps \
  --config-settings=cmake.define.PARAKEET_ROOT="$ROOT" \
  --config-settings=cmake.define.PARAKEET_LIB="$LIB" \
  --config-settings=cmake.define.CMAKE_INSTALL_RPATH="$LIBDIR" \
  --config-settings=cmake.define.CMAKE_BUILD_WITH_INSTALL_RPATH=ON

echo ">> verifying"
python -c "import parakeet_stt; assert parakeet_stt.is_native(), 'still on the stub'; print('native backend:', parakeet_stt.backend_name())"

echo ">> done. run: PARAKEET_MODEL=<model.gguf> pytest"
