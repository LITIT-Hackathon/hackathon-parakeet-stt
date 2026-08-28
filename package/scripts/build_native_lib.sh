#!/usr/bin/env bash
# Build the shared parakeet library the native extension links against.
#
# parakeet.cpp's own build produces a *static* libparakeet.a (compiled without
# -fPIC), which cannot go into a shared object. This rebuilds it position-
# independent and links a single shared libparakeet.so, copying the ggml shared
# libraries alongside it so they resolve via $ORIGIN at load time.
#
# The output is exactly what scripts/wire_native.sh expects:
#   PARAKEET_ROOT = <parakeet.cpp source tree>
#   PARAKEET_LIB  = <out>/libparakeet.so
#
# Usage:
#   scripts/build_native_lib.sh [parakeet.cpp root] [out dir]
# Defaults: ~/parakeet.cpp and <root>/native
set -Eeuo pipefail

ROOT="${1:-$HOME/parakeet.cpp}"
OUT="${2:-$ROOT/native}"

# Pin to the exact commit this package was validated against. parakeet.cpp's
# GGUF schema drifts on master -- the provided .gguf already fails to load on a
# newer build -- so pinning keeps the converter and the runtime in lockstep.
PARAKEET_SHA="e75de9b6b9b688fd293aa22f7e27aa724ea286f8"

if [ ! -d "$ROOT" ]; then
  echo ">> cloning parakeet.cpp into $ROOT"
  git clone --recursive https://github.com/mudler/parakeet.cpp "$ROOT"
fi
echo ">> pinning parakeet.cpp to $PARAKEET_SHA"
git -C "$ROOT" fetch --quiet origin
git -C "$ROOT" checkout --quiet "$PARAKEET_SHA"
git -C "$ROOT" submodule update --init --recursive --quiet

B="$ROOT/build-pic"
echo ">> configuring (PIC) in $B"
cmake -S "$ROOT" -B "$B" \
  -DPARAKEET_BUILD_TESTS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
echo ">> building"
cmake --build "$B" -j

echo ">> linking shared libparakeet.so into $OUT"
mkdir -p "$OUT"
cp -a "$B"/third_party/ggml/src/libggml*.so* "$OUT/"
g++ -shared -fPIC -o "$OUT/libparakeet.so" \
  -Wl,--whole-archive "$B/libparakeet.a" -Wl,--no-whole-archive \
  -L"$OUT" -lggml -lggml-cpu -lggml-base \
  -Wl,-rpath,'$ORIGIN' -pthread

echo ">> done"
echo "PARAKEET_ROOT=$ROOT"
echo "PARAKEET_LIB=$OUT/libparakeet.so"
