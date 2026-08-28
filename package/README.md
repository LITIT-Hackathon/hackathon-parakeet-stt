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

Needs a C++17 compiler, CMake ≥ 3.20, and (on the first build) network access.

```bash
python -m venv .venv && . .venv/bin/activate
pip install .
parakeet info          # -> "backend": "parakeet.cpp", "native": true
```

`pip install` fetches `parakeet.cpp` at a pinned commit and compiles it straight
into the extension — no extra scripts, no separate shared object, everything
static-linked into one `_core` module. First build compiles the engine
(parakeet.cpp + ggml); rebuilds reuse the CMake cache.

You still need a model — see [Model](#model).

### Build knobs

| `pip install . --config-settings=cmake.define.<X>` | effect |
|---|---|
| `PARAKEET_STT_BUNDLED=OFF` | build the canned-text **stub** instead (offline, CI-fast, no engine compile) |
| `PARAKEET_STT_MARCH_NATIVE=OFF` | portable build — drop `-march=native` (needed for a redistributable wheel) |
| `PARAKEET_STT_GIT_TAG=<sha>` | vendor a different `parakeet.cpp` commit |
| `FETCHCONTENT_SOURCE_DIR_PARAKEET_CPP=<path>` | use a local `parakeet.cpp` checkout, skip the clone (offline) |

## Backends

The same Python surface sits on two backends, chosen at build time:

- **native** (default) — `parakeet.cpp` compiled in, real inference.
- **stub** (`PARAKEET_STT_BUNDLED=OFF`) — compiles and returns canned text, so
  the package, CLI, tests, and packaging can be exercised with no engine and no
  model.

## Model

`pip install` does not ship a model. Point `-m` / `PARAKEET_MODEL` at a GGUF
that loads on the pinned `parakeet.cpp`.

> The `parakeet-tdt-0.6b-v3.q8_0.gguf` committed at the repo root is NVIDIA's
> upstream GGUF; parakeet.cpp's GGUF schema has moved since, so it does **not**
> load on the pinned build.

`scripts/build_model.sh` regenerates a matching GGUF from the **provided**
`parakeet-tdt-0.6b-v3.nemo` (same weights, converter and runtime in lockstep):

```bash
git lfs pull --include='parakeet-tdt-0.6b-v3.nemo'
scripts/build_model.sh ../parakeet-tdt-0.6b-v3.nemo ~/parakeet-v3-q8.gguf
PARAKEET_MODEL=~/parakeet-v3-q8.gguf parakeet transcribe audio.wav
```

The conversion pulls the NeMo/torch toolchain (large, one-off). The legacy
`scripts/build_native_lib.sh` + `scripts/wire_native.sh` — building the engine
as a standalone shared library and wiring it in after — still work and are kept
for hacking on the engine, but `pip install` no longer needs them.

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
