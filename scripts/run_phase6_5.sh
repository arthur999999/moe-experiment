#!/usr/bin/env bash
# Phase 6.5 gate: stream ON == stream OFF per dense policy, AND dense policy
# (warm/anon) == mmap baseline with streaming OFF (rebind lossless).
# usage: ./scripts/run_phase6_5.sh [lanes] [dense_modes..]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/phase6_5_stream"
MODEL="$ROOT/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
PROMPT="Explain how mixture of experts routing works."
N_PRED="${N_PRED:-64}"
THREADS="${THREADS:-4}"
SEED="${SEED:-42}"
CACHE_MB="${CACHE_MB:-1024}"
LANES="${1:-4}"
shift || true
DENSE="${@:-0 1 2}"            # 0=mmap 1=warm 2=anon

OUT="$ROOT/outputs"; mkdir -p "$OUT"
name() { case "$1" in 0) echo mmap;; 1) echo warm;; *) echo anon;; esac; }

FAIL=0
off_ok=1   # 1 while every OFF baseline succeeds

for dm in $DENSE; do
  nm=$(name "$dm")
  off_txt="$OUT/gate_off_${nm}.txt"; off_log="$OUT/gate_off_${nm}.log"
  on_txt="$OUT/gate_on_${nm}.txt";   on_log="$OUT/gate_on_${nm}.log"

  echo "== dense=$nm : stream OFF (verify=1) =="
  "$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$CACHE_MB" "$LANES" 0 0 1 "$dm" 0 0 > "$off_txt" 2> "$off_log"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "OFF RUN FAILED (dense=$nm, rc=$rc) — tail of $off_log:"
    tail -n 20 "$off_log"; off_ok=0; FAIL=1
  fi

  echo "== dense=$nm : stream ON  (verify=1) =="
  "$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$CACHE_MB" "$LANES" 0 1 1 "$dm" 0 0 > "$on_txt" 2> "$on_log"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "ON RUN FAILED (dense=$nm, rc=$rc) — CRASH? tail of $on_log:"
    tail -n 20 "$on_log"; FAIL=1
    continue
  fi

  if diff -q "$off_txt" "$on_txt" >/dev/null; then
    echo "GATE PASS ($nm): stream ON == stream OFF"
  else
    echo "GATE FAIL ($nm)"; FAIL=1; diff "$off_txt" "$on_txt" | head -20
  fi
  grep -E 'Phase6_5|Dense:|dense anon boot verify|verify:' "$on_log"
done

# cross-policy: anon/warm stream OFF must equal mmap stream OFF (rebind is lossless)
if [ "$off_ok" -eq 1 ]; then
  base="$OUT/gate_off_mmap.txt"
  for dm in 1 2; do
    nm=$(name "$dm")
    if [ -f "$base" ] && diff -q "$base" "$OUT/gate_off_${nm}.txt" >/dev/null; then
      echo "GATE PASS: dense=$nm (stream OFF) == mmap baseline"
    else echo "GATE FAIL: dense=$nm differs from mmap baseline"; FAIL=1; fi
  done
else
  echo "skip cross-policy check: one or more OFF baselines failed"
fi
exit $FAIL