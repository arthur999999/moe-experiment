#!/usr/bin/env bash
# Phase 5 self-contained gate: stream ON must byte-match stream OFF (resident mmap).
# usage: ./scripts/run_phase5.sh [lanes]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/phase5_stream"
MODEL="$ROOT/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
PROMPT="Explain how mixture of experts routing works."
N_PRED="${N_PRED:-64}"          # short gen: faster gate
THREADS="${THREADS:-4}"
SEED="${SEED:-42}"
CACHE_MB="${CACHE_MB:-1024}"
LANES="${1:-4}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== stream OFF (resident mmap, baseline) =="
"$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$CACHE_MB" "$LANES" 1 0 0 \
  > "$TMP/off.txt" 2> "$TMP/off.log"

echo "== stream ON  (Phase 5 lanes=$LANES) =="
"$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$CACHE_MB" "$LANES" 1 1 1 \
  > "$TMP/on.txt"  2> "$TMP/on.log"

if diff -q "$TMP/off.txt" "$TMP/on.txt" >/dev/null; then
  echo "GATE PASS: stream ON == stream OFF"
else
  echo "GATE FAIL: outputs differ"
  echo "--- diff ---"; diff "$TMP/off.txt" "$TMP/on.txt" | head -40
  exit 1
fi
grep -E 'Phase5|hit_rate|verify' "$TMP/on.log"