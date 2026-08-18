#!/usr/bin/env bash
# Phase 5 benchmark: tok/s vs (cache_mb, lanes) — with optional RAM limiter.
#
# usage:
#   ./scripts/bench_phase5.sh <tokens> <lanes...> -- <caches...>          # no limit
#   ./scripts/bench_phase5.sh -m <budget_mb> <tokens> <lanes...> -- auto  # with limit; cache = budget - overhead
#   ./scripts/bench_phase5.sh -m <budget_mb> <tokens> <lanes...> -- <caches...>
#
# flags:
#   -n <tokens>    number of tokens (or 1st positional arg)
#   -m <budget_mb> enables the limiter: each cell runs via
#                  `systemd-run --scope -p MemoryMax=<budget_mb>M`
#   -o <overhead_mb> fixed overhead for auto mode: dense + staging + kv + base + slack
#                    (OLMoE: dense~160 + staging~260 + kv/base~90 + slack~130 = 640)
#                    PHASE 8: measure again (dense of a 30B is much larger!)
#
# examples:
#   ./scripts/bench_phase5.sh 128 1 2 4 8 -- 0 64 512
#   ./scripts/bench_phase5.sh -m 3000 128 1 2 4 8 -- auto
#   ./scripts/bench_phase5.sh -m 3000 128 1 2 4 8 -- 0 64 512
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/phase5_stream"
MODEL="$ROOT/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
PROMPT="Explain how mixture of experts routing works."
N_PRED="${N_PRED:-128}"
THREADS="${THREADS:-4}"
SEED="${SEED:-42}"
LANES_D=(1 2 4 8)
CACHES_DEFAULT=(0 64 512)
LIMIT_MB=0
OVERHEAD_MB="${OVERHEAD_MB:-640}"

# ---- parse --------------------------------------------------------------
# grammar: [-n <tokens>] [-m <budget_mb>] [-o <overhead_mb>] <tokens> <lanes...> [-- <caches...>]
LANES=(); CACHES=(); TOK_SET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N_PRED="$2"; TOK_SET=1; shift 2 ;;
    -m) LIMIT_MB="$2"; shift 2 ;;
    -o) OVERHEAD_MB="$2"; shift 2 ;;
    --) shift; while [[ $# -gt 0 ]]; do CACHES+=("$1"); shift; done ;;
    *)  if [[ "$TOK_SET" -eq 0 ]]; then N_PRED="$1"; TOK_SET=1;
        else LANES+=("$1"); fi
        shift ;;
  esac
done
[[ ${#LANES[@]} -eq 0 ]] && LANES=("${LANES_DEFAULT[@]}")
[[ ${#CACHES[@]} -eq 0 ]] && CACHES=("${CACHES_DEFAULT[@]}")

# auto mode: cache = budget - overhead (one cell per lane)
if [[ " ${CACHES[*]} " == *" auto "* ]]; then
  if [[ "$LIMIT_MB" -eq 0 ]]; then
    echo "error: cache 'auto' requires -m <budget_mb>" >&2; exit 2
  fi
  AUTO_CACHE=$(( LIMIT_MB - OVERHEAD_MB )); (( AUTO_CACHE < 0 )) && AUTO_CACHE=0
  CACHES=("$AUTO_CACHE")
  echo "== auto cache: budget=${LIMIT_MB}MB - overhead=${OVERHEAD_MB}MB = cache=${AUTO_CACHE}MB"
fi

if command -v systemd-run >/dev/null 2>&1; then :; else
  if [[ "$LIMIT_MB" -gt 0 ]]; then
    echo "warning: systemd-run not found; running WITHOUT limiter" >&2
    LIMIT_MB=0
  fi
fi

echo "== Phase5 bench: n_pred=$N_PRED threads=$THREADS seed=$SEED"
if [[ "$LIMIT_MB" -gt 0 ]]; then
  echo "   LIMITER: MemoryMax=${LIMIT_MB}MB per cell (overhead=${OVERHEAD_MB}MB)"
fi
echo "   caches: ${CACHES[*]} | lanes: ${LANES[*]}"
printf "%-10s %-8s | %-14s | %-8s | %-10s | %-9s | %s\n" cache lanes MemoryMax tok/s hit-rate verify
echo "-----------------------------------------------------------------------"
mkdir -p "$ROOT/outputs"

for mb in "${CACHES[@]}"; do
  for ln in "${LANES[@]}"; do
    log="$ROOT/outputs/bench_t${N_PRED}_c${mb}_l${ln}.log"
    if [[ "$LIMIT_MB" -gt 0 ]]; then
      systemd-run --scope -p "MemoryMax=${LIMIT_MB}M" \
        "$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$mb" "$ln" 1 1 1 \
        >/dev/null 2>"$log"
    else
      "$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$mb" "$ln" 1 1 1 \
        >/dev/null 2>"$log"
    fi
    tok=$(grep -oP '[\d.]+ tok/s' "$log" | head -1)
    hit=$(grep -oP 'hit_rate=[\d.]+%' "$log" | head -1 || echo "hit_rate=—")
    vf=$(grep -oP '\d+ FAILs' "$log" | head -1 || echo "—")
    mm=$([[ "$LIMIT_MB" -gt 0 ]] && echo "${LIMIT_MB}M" || echo "—")
    printf "%-10s %-8s %-14s %-8s %-10s %-9s %s\n" \
           "$mb MB" "$ln" "$mm" "${tok:-—}" "$hit" "$vf"
  done
done