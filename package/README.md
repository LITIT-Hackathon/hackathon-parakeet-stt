# parakeet-stt

[![package CI](https://github.com/LITIT-Hackathon/hackathon-parakeet-stt/actions/workflows/ci.yml/badge.svg?branch=track-b-package)](https://github.com/LITIT-Hackathon/hackathon-parakeet-stt/actions/workflows/ci.yml)

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

### Build native from the provided model

Three scripts take a clean checkout to a working native install. Run them from
`package/` (Linux; needs a C++17 compiler, CMake, and — for the conversion —
enough disk for the NeMo/torch toolchain):

```bash
# 1. build parakeet.cpp at the pinned commit and link the shared library
scripts/build_native_lib.sh ~/parakeet.cpp ~/parakeet.cpp/native

# 2. convert the PROVIDED .nemo into a GGUF that loads on that build
#    (first: git lfs pull --include='parakeet-tdt-0.6b-v3.nemo')
scripts/build_model.sh ../parakeet-tdt-0.6b-v3.nemo ~/parakeet-v3-q8.gguf ~/parakeet.cpp

# 3. reinstall the package against the native backend
scripts/wire_native.sh ~/parakeet.cpp ~/parakeet.cpp/native/libparakeet.so

parakeet info                 # -> "backend": "parakeet.cpp"
PARAKEET_MODEL=~/parakeet-v3-q8.gguf pytest
```

Under the hood step 3 is just `pip install .` with
`--config-settings=cmake.define.PARAKEET_ROOT=… PARAKEET_LIB=…`; the script adds
the rpath so the extension resolves the library at import.

> **On the provided `.gguf`.** The `parakeet-tdt-0.6b-v3.q8_0.gguf` committed at
> the repo root is NVIDIA's upstream GGUF. parakeet.cpp's GGUF schema has moved
> since it was published, so it does **not** load on the pinned build.
> `build_model.sh` regenerates a matching GGUF from the provided `.nemo` — same
> weights, converter and runtime in lockstep — so point `-m` / `PARAKEET_MODEL`
> at that regenerated file, not the committed one.

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

## Benchmark

Thread-pinned, end-to-end (mel + encoder + decode); the one-off model load
(~0.41 s) is measured separately and excluded. `q8_0` model, GCP
`n2-standard-16` (Cascade Lake), via `parakeet-cli bench`.

| threads | RTF, English (7.4 s) | RTF, German (12 s) |
|--------:|:--------------------:|:------------------:|
|       1 |         0.27         |        0.28        |
|       2 |         0.15         |        0.15        |
|       4 |        0.083         |       0.082        |
|       8 |        0.049         |       0.048        |

RTF is transcription time / audio duration; below 1.0 is faster than realtime.
Even a single core runs ~3.6x faster than realtime, so this holds comfortably
real-time on a laptop — pin threads (or state the hardware) when you quote a
number, since a many-core box flatters it.

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
