# parakeet-stt

[![package CI](https://github.com/LITIT-Hackathon/hackathon-parakeet-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/LITIT-Hackathon/hackathon-parakeet-stt/actions/workflows/ci.yml)

Local speech-to-text on NVIDIA Parakeet with a native C++ inference core,
exposed as an installable Python package with a CLI.

```python
from parakeet_stt import Model

with Model("parakeet-v3-q8.gguf") as m:        # see Model, below
    result = m.transcribe("audio.wav")

print(result.text)
print(result.rtf, result.latency_ms)      # runtime metrics come with the transcript
```

## Install

Linux x86_64 for now. The native engine is compiled from source, so a build
needs a C++17 compiler, CMake ≥ 3.20, and (on the first build) network access.
Prebuilt wheels, macOS, and Windows are future work; see
[Known limitations](#known-limitations).

```bash
python -m venv .venv && . .venv/bin/activate
pip install .
parakeet download-model       # ~940 MB into a local cache, once
parakeet transcribe audio.wav
```

`pip install` fetches `parakeet.cpp` at a pinned commit and compiles it straight
into the extension — no extra scripts, no separate shared object, everything
static-linked into one `_core` module. First build compiles the engine
(parakeet.cpp + ggml); rebuilds reuse the CMake cache.

`parakeet transcribe` downloads the default model on first use if it is not
already cached, so `download-model` is optional — see [Model](#model).

### Build knobs

| `pip install . --config-settings=cmake.define.<X>` | effect |
|---|---|
| `PARAKEET_STT_BUNDLED=OFF` | build the canned-text **stub** instead (offline, CI-fast, no engine compile) |
| `PARAKEET_STT_MARCH_NATIVE=ON` | let ggml use `-march=native` (~30% faster); off by default so the build stays portable |
| `PARAKEET_STT_GIT_TAG=<sha>` | vendor a different `parakeet.cpp` commit |
| `FETCHCONTENT_SOURCE_DIR_PARAKEET_CPP=<path>` | use a local `parakeet.cpp` checkout, skip the clone (offline) |

## Backends

The same Python surface sits on two backends, chosen at build time:

- **native** (default) — `parakeet.cpp` compiled in, real inference.
- **stub** (`PARAKEET_STT_BUNDLED=OFF`) — compiles and returns canned text, so
  the package, CLI, tests, and packaging can be exercised with no engine and no
  model.

## Model

`pip install` does not ship weights. A model is referenced by a short name and
fetched into a local cache (`platformdirs` cache dir, or `PARAKEET_CACHE_DIR`)
on first use, with its SHA-256 checked.

```bash
parakeet list-models
parakeet download-model                 # the default, parakeet-tdt-0.6b-v3
parakeet transcribe a.wav               # downloads on first use if needed
parakeet transcribe a.wav -m ~/my.gguf  # or point at a local file
```

```python
Model()                      # default model, cached on first use
Model("parakeet-tdt-0.6b-v3")  # same, by name
Model("/path/to/model.gguf")   # explicit file, no download
```

> The `parakeet-tdt-0.6b-v3.q8_0.gguf` committed at the repo root is NVIDIA's
> upstream GGUF; parakeet.cpp's GGUF schema has moved since, so it does **not**
> load on the pinned build. The registry points at a converted copy.

### Regenerating the model

`scripts/build_model.sh` produces the registry's GGUF from the **provided**
`parakeet-tdt-0.6b-v3.nemo` (same weights; converter and runtime pinned to the
same commit). Self-contained: it fetches `parakeet.cpp` for the converter and
verifies the result by loading it through the installed `parakeet_stt`.

```bash
git lfs pull --include='parakeet-tdt-0.6b-v3.nemo'
scripts/build_model.sh ../parakeet-tdt-0.6b-v3.nemo ~/parakeet-v3-q8.gguf
PARAKEET_MODEL=~/parakeet-v3-q8.gguf parakeet transcribe audio.wav
```

The conversion pulls the NeMo/torch toolchain (large, one-off). Only whoever
publishes the release asset runs this; users get the cached download.

### Offline install

`pip install` needs network once, to fetch `parakeet.cpp`. To build with no
network, clone it yourself and point the build at the checkout:

```bash
git clone https://github.com/mudler/parakeet.cpp && \
  git -C parakeet.cpp checkout e75de9b6b9b688fd293aa22f7e27aa724ea286f8 && \
  git -C parakeet.cpp submodule update --init --recursive
pip install . --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_PARAKEET_CPP="$PWD/parakeet.cpp"
```

## Use

```bash
# transcript to stdout, metrics to stderr
parakeet transcribe audio.wav -m ~/parakeet-v3-q8.gguf

# full result as JSON
parakeet transcribe audio.wav -m ~/parakeet-v3-q8.gguf --json

# model path from the environment
export PARAKEET_MODEL=~/parakeet-v3-q8.gguf
parakeet transcribe audio.wav
```

Any sample rate or channel count is accepted; input is normalised to 16 kHz
mono before inference. Feeding the wrong rate is the classic silent failure, so
that conversion is done for you and not left to the caller.

## Metrics and word timings

Every `transcribe()` returns a `Result`:

| field | meaning |
|---|---|
| `text` | the full transcript |
| `words` | tuple of `Word(text, start, end, conf)` -- per-word spans in seconds and confidence in (0, 1]; empty on the stub backend |
| `audio_s` | length of the input audio in seconds |
| `latency_ms` | wall clock for the transcription call alone |
| `rtf` | latency / audio duration; below 1.0 is faster than realtime |
| `load_ms` | one-off model load, measured separately from inference |
| `backend` | `parakeet.cpp` or `stub` |

`parakeet transcribe a.wav --json` emits all of it, `words` as a list of
`{"text","start","end","conf"}`.

Report RTF with the thread count and hardware stated. A number from a many-core
cloud box is not the number a "lightweight local runtime" delivers on a laptop,
so pin threads (4-8) or take finals on the target hardware.

## Benchmark

Thread-pinned, end-to-end (mel + encoder + decode); the one-off model load
(~0.41 s) is measured separately and excluded. `q8_0` model, GCP
`n2-standard-16` (Cascade Lake), via `parakeet-cli bench` from a standalone
`parakeet.cpp` build (the bundled install does not build the CLI).

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
pytest                                    # native build; model-dependent tests skip
PARAKEET_MODEL=~/parakeet-v3-q8.gguf pytest   # runs the full native tier
```

## Reproduce from a clean clone

```bash
git clone <repo-url> && cd hackathon-parakeet-stt/package
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest
parakeet info
```

## Known limitations

- **Linux x86_64 only.** The extension is built from source; there are no
  prebuilt wheels yet, and the macOS / Windows build paths are untested.
- **Model is fetched, not bundled.** ~940 MB on first use, from a GitHub release
  asset. `PARAKEET_CACHE_DIR` relocates the cache; `Model(..., download=False)`
  refuses to fetch.
- **CPU only.** No CUDA / Metal / Vulkan.
- **Batch of one.** No batched or streaming transcription API (the engine has a
  batch path; the package calls it with one clip).
