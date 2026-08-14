#!/usr/bin/env bash
# Builds the Phase 3 streamer against the already compiled llama.cpp (build/bin).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="$ROOT/llama.cpp"
BUILD="$LLAMA/build/bin"
SRC="$ROOT/src/cxx/phase3_stream.cpp"
OUT="$ROOT/src/cxx/phase3_stream"

g++ -std=c++11 -O2 \
  -I "$LLAMA/include" -I "$LLAMA/ggml/include" \
  "$SRC" \
  -L "$BUILD" -o "$OUT" \
  -lllama -lggml -lggml-base -lggml-cpu -lpthread -lm -ldl \
  -Wl,-rpath,"$BUILD"

echo "build: $OUT"