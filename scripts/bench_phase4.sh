#!/usr/bin/env bash
# Phase 4 benchmark: measures the REAL RAM COST of streaming.
#
# Splits the process memory into:
#   RssAnon = anonymous RAM allocated by YOUR code
#             (LRU cache + compact buffers + KV + heap)
#   RssFile = file-backed RAM (model mmap pages resident)
#
# What this answers:
#   "how much RAM costs to load ONLY the cache + experts in use?"
#   => RssAnon peak (that's what your driver pays beyond the file)
#
# with model > RAM (Phase 8), RssFile doesn't stay — only Anon matters.
#
# uso: ./scripts/bench_phase4.sh [-n tokens] [cache_mb ...]
#   (default n_pred=128, cache: 0 64 512 1024 2048)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/phase4_stream"
MODEL="$ROOT/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
PROMPT="Explain how mixture of experts routing works."
N_PRED="${N_PRED:-128}"
THREADS="${THREADS:-4}"
SEED="${SEED:-42}"

# argument parsing: -n <tokens> antes dos tamanhos de cache
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) N_PRED="$2"; shift 2 ;;
        *)  ARGS+=("$1"); shift  ;;
    esac
done
CACHES=("${ARGS[@]:-0 64 512 1024 2048}")
OUT_DIR="$ROOT/outputs/phase4"
mkdir -p "$OUT_DIR"

printf "%-16s | %-8s | %-14s | %-14s | %-10s | %s\n" \
       "cache" "tok/s" "RssAnon peak" "RssFile peak" "hit" "disk"
echo "--+---------------------------------------------------------------------------------"

for mb in "${CACHES[@]}"; do
    run="$OUT_DIR/bench_m${mb}.log"

    "$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$mb" >/dev/null 2>"$run" &
    pid=$!

    peak_anon=0; peak_file=0
    # monitor /proc until process finishes
    while kill -0 "$pid" 2>/dev/null; do
        anon=$(awk '/^RssAnon:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
        file=$(awk '/^RssFile:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
        [ "${anon:-0}" -gt "$peak_anon" ] && peak_anon=$anon
        [ "${file:-0}" -gt "$peak_file" ] && peak_file=$file
        sleep 0.03
    done
    wait "$pid" || true

    tok_s=$(grep -oP '[\d.]+ tok/s' "$run" | head -1)
    hit=$(grep -oP 'hit_rate=[\d.]+%' "$run" | head -1 || true)
    reads=$(grep -oP 'disk_reads=\d+' "$run" | head -1 || true)

    printf "cache=%-10s %-8s %-14s %-14s %-10s %s\n" \
           "$mb MB" "${tok_s:-—}" "$peak_anon kB" "$peak_file kB" "${hit:-—}" "$reads"
done

echo
echo "Interpretation:"
echo "  RssAnon(cache=1024) - RssAnon(cache=0) ≈ incremental cache cost (MB)"
echo "  RssAnon(cache=0)                        ≈ base streaming cost (buffers + KV + heap)"
echo "  With model > RAM (Phase 8), RssAnon is what matters — the OS doesn't keep the entire file resident."