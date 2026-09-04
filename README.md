# Parakeet Native STT

Build a lightweight local Speech-to-Text runtime around NVIDIA Parakeet,
using a native C++ inference core and exposing the result as a simple,
installable Python library.

## Presentation

[![View on Figma](https://img.shields.io/badge/Figma-Presentation-a259ff?logo=figma&logoColor=white)](https://www.figma.com/deck/wQH7IdCeAYWV86n7YwwIoo)

## MVP

Your solution should:

- Load the provided Parakeet model.
- Accept WAV/PCM audio input.
- Perform transcription through native C++ code.
- Expose the native functionality to Python.
- Be installable as a Python package.
- Provide a simple transcription API.
- Provide a CLI for verification.
- Return transcript text and basic runtime metrics.
- Include a repeatable smoke test.

Example target API:

    model.transcribe("audio.wav")

## Provided

- Parakeet reference implementation/repository https://github.com/mudler/parakeet.cpp
- Parakeet model assets .nemo and .gguf
- English and German test audio: German -> https://learngerman.dw.com/de/langsam-gesprochene-nachrichten/s-60040332 and English: https://www.bbc.com/audio/play/p0p6c0k7
- GCP sandbox and token credits

## Out of Scope

Training or fine-tuning the model is not required.

Real-time streaming, word-level timestamps and broader hardware
optimization are considered stretch goals.

## Deliverable

A working and demonstrable MVP with reproducible setup instructions.
