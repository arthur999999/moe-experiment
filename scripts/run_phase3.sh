#!/usr/bin/env bash
# Runs the Phase 3 streamer with fixed config and separates stdout (text) / stderr (logs).
# usage: ./scripts/run_phase3.sh [n_pred] [threads] [seed]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/src/cxx/phase3_stream"
MODEL="$ROOT/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
PROMPT="Explain how mixture of experts routing works."
N_PRED="${1:-128}"
THREADS="${2:-4}"
SEED="${3:-42}"

OUT_DIR="$ROOT/outputs/phase3"
mkdir -p "$OUT_DIR"

"$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" \
  > "$OUT_DIR/phase3_output.txt" \
  2> "$OUT_DIR/phase3_stderr.txt"

echo "   run complete"
echo "   texto:  $OUT_DIR/phase3_output.txt"
echo "   logs:   $OUT_DIR/phase3_stderr.txt"
grep -E "data_offset|warm-up|Phase 3" "$OUT_DIR/phase3_stderr.txt"