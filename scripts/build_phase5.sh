#!/usr/bin/env bash
# Build the Phase 5 streamer against the already-compiled llama.cpp.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="$ROOT/llama.cpp"
BUILD="$LLAMA/build/bin"
SRC="$ROOT/src/cxx/phase5_stream.cpp"
OUT="$ROOT/build/phase5_stream"

mkdir -p "$ROOT/build"

g++ -std=c++11 -O2 -pthread \
  -I "$LLAMA/include" -I "$LLAMA/ggml/include" \
  "$SRC" \
  -L "$BUILD" -o "$OUT" \
  -lllama -lggml -lggml-base -lggml-cpu -lpthread -lm -ldl \
  -Wl,-rpath,"$BUILD"

echo "OK: $OUT"