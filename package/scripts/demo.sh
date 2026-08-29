#!/usr/bin/env bash
# One-command demo of the deliverable: a lightweight local STT runtime around
# NVIDIA Parakeet, native C++ core, exposed as an installable Python library.
#
# Each section proves one clause of that sentence, using the bundled test
# fixtures (no external audio needed).
#
#   scripts/demo.sh [model.gguf]
#
# Model resolution: the argument, else $PARAKEET_MODEL, else the default model
# via `parakeet download-model`. Run from an environment where the package is
# installed (`pip install -e .` from package/).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # package/
EN="$ROOT/tests/fixtures/speech_en.wav"
DE="$ROOT/tests/fixtures/speech_de.wav"

bold() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

bold "1. installable Python library"
if ! python3 -c "import parakeet_stt" 2>/dev/null; then
  echo "parakeet_stt is not installed. From the package/ directory:" >&2
  echo "    python -m venv .venv && . .venv/bin/activate && pip install -e ." >&2
  exit 1
fi
python3 -c "import parakeet_stt as p; print('parakeet-stt', p.__version__, '| native:', p.is_native())"

# -- resolve a model --------------------------------------------------------
MODEL="${1:-${PARAKEET_MODEL:-}}"
if [ -z "$MODEL" ] && command -v parakeet >/dev/null 2>&1 \
     && parakeet list-models >/dev/null 2>&1; then
  echo ">> no model given; fetching the default with 'parakeet download-model'"
  MODEL="$(parakeet download-model)" || MODEL=""
fi
if [ -z "${MODEL:-}" ] || [ ! -f "$MODEL" ]; then
  {
    echo "No model available. Give one of:"
    echo "    scripts/demo.sh /path/to/model.gguf"
    echo "    PARAKEET_MODEL=/path/to/model.gguf scripts/demo.sh"
    echo "  or build it from the provided .nemo:"
    echo "    scripts/build_model.sh ../parakeet-tdt-0.6b-v3.nemo ~/parakeet-v3-q8.gguf"
  } >&2
  exit 1
fi
echo "model: $MODEL"

bold "2. native C++ inference core"
parakeet info

bold "3. simple library API: model.transcribe() + word timings"
python3 - "$MODEL" "$EN" <<'PY'
import sys
from parakeet_stt import Model
model_path, wav = sys.argv[1], sys.argv[2]
with Model(model_path) as m:
    r = m.transcribe(wav)
print("text   :", r.text)
print("metrics: audio %.2fs  load %.0fms  infer %.0fms  RTF %.3f  backend %s"
      % (r.audio_s, r.load_ms, r.latency_ms, r.rtf, r.backend))
words = getattr(r, "words", None) or ()
if words:
    print("words  :", "  ".join(f"{w.text}@{w.start:.2f}s" for w in words[:6]), "...")
PY

bold "4. CLI + runtime metrics (English)"
parakeet transcribe "$EN" -m "$MODEL" --json

bold "5. local + multilingual Parakeet (German, same model)"
parakeet transcribe "$DE" -m "$MODEL"

bold "6. repeatable smoke test"
PARAKEET_MODEL="$MODEL" python3 -m pytest -q "$ROOT/tests/test_smoke.py"

bold "done"
echo "Lightweight local STT on NVIDIA Parakeet, native C++ core, Python library - all shown above."
