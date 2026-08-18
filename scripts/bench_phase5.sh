#!/usr/bin/env bash
# Phase 5 benchmark: tok/s vs (cache_mb, lanes) — with RAM limiter + cgroup-aware memory monitoring.
#
# usage:
#   ./scripts/bench_phase5.sh <tokens> <lanes...> -- <caches...>          # no limit
#   ./scripts/bench_phase5.sh -m <budget_mb> <tokens> <lanes...> -- auto  # with limit; cache = budget - overhead
#   ./scripts/bench_phase5.sh -m <budget_mb> <tokens> <lanes...> -- <caches...>
#
# flags:
#   -n <tokens>       number of tokens (or 1st positional arg)
#   -m <budget_mb>    enables the limiter: systemd-run -p MemoryMax= (v2) / MemoryLimit= (v1)
#   -o <overhead_mb>  fixed overhead for auto mode (OLMoE default 640; PHASE 8: re-measure)
#   -k                KEEP page cache between cells (default: drop before each cell)
#
# WHY we drop the page cache per cell (CRITICAL under cgroup v2):
#   File pages are charged to the cgroup that FAULTS them in. If the model's
#   pages are already resident (charged to another cgroup) from a previous run,
#   a limited scope reading them pays NOTHING -> memory.max never bites -> no
#   reclaim, majflt=0, MaxRSS balloons past the budget (your symptom:
#   MemoryMax=700M + MaxRSS=4394MB). Dropping per cell makes the cell's cgroup
#   fault the file cold so memory.current genuinely grows toward the cap.
#
# memory columns:
#   mem_peak_MB  cgroup memory.peak (authoritative — what the scope actually paid)
#   maxrss_MB    process peak RSS (/usr/bin/time -v; pages charged elsewhere show
#                up here but NOT in mem_peak — under a limiter trust mem_peak)
#   majflt       major page faults = reclaimed pages re-read from disk (thrash)
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
DROP_CACHE=1                       # default: drop page cache before each cell

TIME_BIN=""
if [[ -x /usr/bin/time ]]; then
  TIME_BIN=/usr/bin/time
else
  echo "warning: /usr/bin/time not found (sudo apt install time) — RSS/fault columns will show '?'" >&2
fi

# ---- parse --------------------------------------------------------------
LANES=(); CACHES=(); TOK_SET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N_PRED="$2"; TOK_SET=1; shift 2 ;;
    -m) LIMIT_MB="$2"; shift 2 ;;
    -o) OVERHEAD_MB="$2"; shift 2 ;;
    -k) DROP_CACHE=0; shift ;;
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

# ---- cgroup capability --------------------------------------------------
CGROUP_FS="$(stat -fc %T /sys/fs/cgroup 2>/dev/null || echo unknown)"
if [[ "$CGROUP_FS" == "cgroup2fs" ]]; then
  MEM_PROPS_FMT="MemoryMax=%dM"
else
  MEM_PROPS_FMT="MemoryLimit=%dM"
  echo "note: cgroup v1 detected — using MemoryLimit (MemoryMax is ignored on v1)" >&2
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  if [[ "$LIMIT_MB" -gt 0 ]]; then
    echo "warning: systemd-run not found; running WITHOUT limiter" >&2
    LIMIT_MB=0
  fi
fi

if [[ "$LIMIT_MB" -gt 0 && "$(id -u)" -ne 0 ]]; then
  echo "warning: not root — cannot drop page cache / enforce MemoryMax reliably; results may be polluted" >&2
fi

echo "== Phase5 bench: n_pred=$N_PRED threads=$THREADS seed=$SEED"
echo "   cgroup: $CGROUP_FS | drop page cache per cell: $([ "$DROP_CACHE" -eq 1 ] && echo ON || echo OFF)"
if [[ "$LIMIT_MB" -gt 0 ]]; then
  printf -v MEM_PROPS "$MEM_PROPS_FMT" "$LIMIT_MB"
  echo "   LIMITER: $MEM_PROPS (overhead=${OVERHEAD_MB}MB)"
fi
echo "   model size: $(du -h "$MODEL" | cut -f1)"
echo "   caches: ${CACHES[*]} | lanes: ${LANES[*]}"
printf "%-10s %-8s | %-14s | %-8s | %-10s | %-9s | %-10s | %-8s | %-8s | %s\n" \
  cache lanes MemoryMax tok/s hit-rate verify mem_peak_MB maxrss_MB majflt
echo "----------------------------------------------------------------------------------------------------------------"
mkdir -p "$ROOT/outputs"

for mb in "${CACHES[@]}"; do
  for ln in "${LANES[@]}"; do
    log="$ROOT/outputs/bench_t${N_PRED}_c${mb}_l${ln}.log"
    rm -f "$log"

    # CRITICAL: drop the page cache so THIS cell's cgroup pays for the pages
    if [[ "$LIMIT_MB" -gt 0 && "$DROP_CACHE" -eq 1 && "$(id -u)" -eq 0 ]]; then
      sync >/dev/null 2>&1 || true
      echo 3 > /proc/sys/vm/drop_caches
    fi

    # args: model prompt n_pred threads seed cache_mb lanes dontneed(0=off) stream verify
    inner=("$BIN" "$MODEL" "$PROMPT" "$N_PRED" "$THREADS" "$SEED" "$mb" "$ln" 1 1 1)
    if [[ -n "$TIME_BIN" ]]; then
      inner=("$TIME_BIN" -v "${inner[@]}")
    fi

    # ---- run the cell ---------------------------------------------------
    if [[ "$LIMIT_MB" -gt 0 ]]; then
      systemd-run --scope -p "$MEM_PROPS" \
        "${inner[@]}" >/dev/null 2>"$log" &
      SR_PID=$!

      # find the scope's cgroup via the newest phase5_stream process
      BP=""
      for _ in $(seq 1 20); do
        BP=$(pgrep -n -f 'phase5_stream' 2>/dev/null || true)
        [[ -n "$BP" ]] && break
        sleep 0.1
      done
      CG=""
      if [[ -n "$BP" ]]; then
        CG=$(awk -F: '{print $3}' "/proc/$BP/cgroup" 2>/dev/null || true)
      fi

      # poll memory.peak (monotonic high-water mark) while the scope lives
      PEAK="?"
      if [[ -n "$CG" && -r "/sys/fs/cgroup${CG}/memory.peak" ]]; then
        while kill -0 "$SR_PID" 2>/dev/null; do
          PEAK=$(cat "/sys/fs/cgroup${CG}/memory.peak" 2>/dev/null || echo "$PEAK")
          sleep 0.2
        done
        PEAK=$(cat "/sys/fs/cgroup${CG}/memory.peak" 2>/dev/null || echo "$PEAK")
      fi
      wait "$SR_PID" || true
      if [[ "$PEAK" != "?" ]]; then
        mem_peak=$(( PEAK / 1024 / 1024 ))
      else
        mem_peak="?"
      fi
    else
      "${inner[@]}" >/dev/null 2>"$log"
      mem_peak="—"
    fi

    # ---- parse results ---------------------------------------------------
    tok=$(grep -oP '[\d.]+ tok/s' "$log" | head -1)
    hit=$(grep -oP 'hit_rate=[\d.]+%' "$log" | head -1 || echo "hit_rate=—")
    vf=$(grep -oP '\d+ FAILs' "$log" | head -1 || echo "—")
    mm=$([[ "$LIMIT_MB" -gt 0 ]] && echo "${LIMIT_MB}M" || echo "—")

    maxrss_kb=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' "$log" || echo "")
    maxrss_mb=$([[ -n "$maxrss_kb" ]] && echo $(( maxrss_kb / 1024 )) || echo "?")
    majflt=$(grep -oP 'Major \(requiring I/O\) page faults: \K[0-9]+' "$log" || echo "?")

    printf "%-10s %-8s %-14s %-8s %-10s %-9s %-10s %-8s %-8s\n" \
           "$mb MB" "$ln" "$mm" "${tok:-—}" "$hit" "$vf" "$mem_peak" "$maxrss_mb" "$majflt"
  done
done