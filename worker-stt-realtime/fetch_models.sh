#!/bin/sh
# Populate docker_models/ (int8 zipformer + silero VAD) from public sherpa-onnx release assets.
set -e; cd "$(dirname "$0")"; M=sherpa-onnx-streaming-zipformer-en-2023-06-26; B=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
mkdir -p docker_models tmp_fetch && curl -fsSL $B/$M.tar.bz2 | tar xj -C tmp_fetch
mkdir -p docker_models/$M && cp tmp_fetch/$M/*int8.onnx tmp_fetch/$M/tokens.txt docker_models/$M/ && rm -rf tmp_fetch
curl -fsSL -o docker_models/silero_vad.onnx $B/silero_vad.onnx
