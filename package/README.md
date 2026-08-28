# parakeet-stt

Local speech-to-text on NVIDIA Parakeet with a native C++ inference core,
exposed as an installable Python package with a CLI.

```python
from parakeet_stt import Model

with Model("parakeet-tdt-0.6b-v3.gguf") as m:
    result = m.transcribe("audio.wav")

print(result.text)
print(result.rtf, result.latency_ms)      # runtime metrics come with the transcript
```

## Install

Building the extension needs a C++17 compiler and CMake. The package installs
either against the real engine or against a built-in stub, decided at build
time (see below).

```bash
python -m venv .venv && . .venv/bin/activate
pip install .
```

Verify the build:

```bash
parakeet info
```

## Backends

The same Python surface sits on top of two backends:

- **stub** (default) compiles and returns canned text. It exists so the whole
  package, CLI, tests, and packaging work before the native engine is wired in.
- **native** links `parakeet.cpp` and runs real inference.

Build native by pointing the build at a `parakeet.cpp` source tree and its
built shared library:

```bash
pip install . \
  --config-settings=cmake.define.PARAKEET_ROOT=/opt/parakeet.cpp \
  --config-settings=cmake.define.PARAKEET_LIB=/opt/parakeet.cpp/build-shared/libparakeet.so
```

`scripts/wire_native.sh` does this end to end on a machine that already has the
source and the library.

## Use

```bash
# transcript to stdout, metrics to stderr
parakeet transcribe audio.wav -m parakeet-tdt-0.6b-v3.gguf

# full result as JSON
parakeet transcribe audio.wav -m model.gguf --json

# model path from the environment
export PARAKEET_MODEL=parakeet-tdt-0.6b-v3.gguf
parakeet transcribe audio.wav
```

Any sample rate or channel count is accepted; input is normalised to 16 kHz
mono before inference. Feeding the wrong rate is the classic silent failure, so
that conversion is done for you and not left to the caller.

## Metrics

Every `transcribe()` returns the transcript together with:

| field | meaning |
|---|---|
| `audio_s` | length of the input audio in seconds |
| `latency_ms` | wall clock for the transcription call alone |
| `rtf` | latency / audio duration; below 1.0 is faster than realtime |
| `load_ms` | one-off model load, measured separately from inference |
| `backend` | `parakeet.cpp` or `stub` |

Report RTF with the thread count and hardware stated. A number from a many-core
cloud box is not the number a "lightweight local runtime" delivers on a laptop,
so pin threads (4-8) or take finals on the target hardware.

## Test

```bash
pip install -e '.[test]'
pytest                                    # audio + packaging tests, stub backend
PARAKEET_MODEL=model.gguf pytest          # adds real end-to-end inference
```

## Reproduce from a clean clone

```bash
git clone <url> && cd parakeet-stt
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest
parakeet info
```
