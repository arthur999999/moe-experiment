#!/usr/bin/env bash
# bench_phase6_5_ram.sh — memory-focused bench (Phase 6.5), 4 lanes.
# Two configs, both stream=1 (verify=0):
#   A: dense=2 (anon) + dontneed=1   -> "no page-cache copies"
#   B: dense=0 (mmap) + dontneed=0   -> "resident baseline"
# Reports peak RAM (VmRSS) + RssAnon/RssFile split + tok/s + purity + cache stats.
# usage:
#   CACHE_MB="0 512 1024" N_PRED="128" ./scripts/bench_phase6_5_ram.sh
# env: MODEL PROMPT CACHE_MB N_PRED LANES THREADS SEED DROP(S=1 flush between cfgs, needs root)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/phase6_5_stream"
MODEL="$ROOT/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
PROMPT="${PROMPT:-Explain how mixture of experts routing works.}"
LANES="${LANES:-4}"; THREADS="${THREADS:-4}"; SEED="${SEED:-42}"
CACHE_MB="${CACHE_MB:-0 512 1024}"
N_PRED="${N_PRED:-128}"
SENSOR=32
DROP="${DROP:-0}"
DENSE_ON_MODE="${DENSE_ON_MODE:-2}"   # 2=anon
DENSE_OFF_MODE="${DENSE_OFF_MODE:-0}" # 0=mmap
mkdir -p "$ROOT/outputs"

# one config run; args: short dense_mode dontneed cache npred
run_cfg() {
  local short="$1" dm="$2" dn="$3" cache="$4" npred="$5"
  local log="$ROOT/outputs/ram_${short}_c${cache}_n${npred}.log"
  local out="$ROOT/outputs/ram_${short}_c${cache}_n${npred}.txt"
  "$BIN" "$MODEL" "$PROMPT" "$npred" "$THREADS" "$SEED" "$cache" "$LANES" \
    "$dn" 1 0 "$dm" 0 "$SENSOR" >"$out" 2>"$log" &
  local pid=$!
  local peak_rss=0 peak_anon=0 peak_file=0
  # /proc sampler: track peaks of VmRSS / RssAnon / RssFile (kB
  while kill -0 "$pid" 2>/dev/null; do
    IFS=' ' read -r r a f <<< "$(awk '/^VmRSS:/{r=$2}/^RssAnon:/{a=$2}/^RssFile:/{f=$2}END{print r+0,a+0,f+0}' /proc/$pid/status 2>/dev/null)"
    (( r > peak_rss )) && peak_rss=$r
    (( a > peak_anon )) && peak_anon=$a
    (( f > peak_file )) && peak_file=$f
    sleep 0.05
  done
  wait "$pid"; local rc=$?
  local tok=$(grep -oP '[\d.]+ tok/s' "$log" | tail -1)
  local dres=$(grep '^Dense:' "$log" | grep -oP 'resident_frac=\K-?[\d.]+')
  local hit=$(grep -oP 'hit_rate=\K[\d.]+' "$log" | tail -1)
  [ -n "$hit" ] || hit="-"
  local reads=$(grep -oP 'disk_reads=\K[0-9]+' "$log" | tail -1); reads=${reads:-0}
  local maj=$(grep -oP 'majflt_total=\K[0-9]+' "$log" | tail -1); maj=${maj:-0}
  [ "$rc" -eq 0 ] || tok="FAIL($rc)"
  printf "%-10s %-6s %-5s | %-8s %-8s %-8s %-8s %-7s %-6s %-8s %-8s\n" \
    "$short" "$cache" "$npred" "$tok" "$((peak_rss/1024))" "$((peak_anon/1024))" \
    "$((peak_file/1024))" "$dres" "$hit" "$reads" "$maj"
}

printf "== Phase 6.5 RAM bench (lanes=%d threads=%d verify=0) ==\n" "$LANES" "$THREADS"
printf "   A: stream=1 dense=anon dontneed=1    B: stream=1 dense=mmap dontneed=0\n"
printf "%-10s %-6s %-5s | %-8s %-8s %-8s %-8s %-7s %-6s %-8s %-8s\n" \
  "cfg" "cacheMB" "tok" "tok/s" "RSS_peak" "RssAnon" "RssFile" "dense%" "hit%" "reads" "majflt"
printf -- "----------------------------------------------------------------------\n"

for npred in $N_PRED; do
  for cache in $CACHE_MB; do
    [ "$DROP" = "1" ] && { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true; }
    run_cfg "anon_dn1" "$DENSE_ON_MODE"  1 "$cache" "$npred"
    [ "$DROP" = "1" ] && { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true; }
    run_cfg "mmap_dn0" "$DENSE_OFF_MODE" 0 "$cache" "$npred"
  done
done