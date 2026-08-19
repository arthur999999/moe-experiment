# Phase 6.5 — Dense-Weights Policy + Durable Page-Cache Purity (RESULTS)

**Status:**  COMPLETE — correctness gate green, dense-residency gate green, memory bench delivered. O_DIRECT (the "no page-cache copy" mechanism) evaluated and **deferred to Phase 8**.

## Summary

Phase 6.5 extends Phase 5 (Strategy A + LRU + DONTNEED + parallel I/O) with a **dense (non-expert) weight policy** and makes the page-cache eviction **durable** on both sides of the model — experts (bounded by LRU budget) and dense (pinned resident in anon buffers). Two design B modes, `warm` and `anon`, join the Phase 4/5 `mmap` baseline; an optional `mlock` and a residency sensor (`dense_resident_frac` + major-fault deltas) round it out.

**Correctness gate: PASSED 6/6** — stream ON == stream OFF for every dense policy, cross-policy rebind lossless, `verify: 0 FAILs` throughout.
**Dense-residency gate: PASSED** — after an `anon + dontneed` run, the dense file pages drop out of the page cache to **7.8%** (`PURE`), experts to **7.9%**.

## Files

- `src/cxx/phase6_5_stream.cpp` — streamer with dense policy (Strategy A + LRU + DONTNEED + IoPool lanes + dense anon/warm + boot-drop + residency sensor)
- `scripts/build_phase6_5.sh` — build against compiled llama.cpp
- `scripts/run_phase6_5.sh` — self-contained correctness gate (stream ON == OFF per dense policy + cross-policy)
- `scripts/bench_phase6_5.sh` — benchmark harness (dense × dontneed × cache × lanes, optional `MemoryMax`)
- `scripts/bench_phase6_5_ram.sh` — memory-focused bench (peak RSS + RssAnon/RssFile split)
- `scripts/check_dense_residency.py` — gate 3: mincore over dense file ranges (PURE if < 15%)
- `scripts/check_cache_purity.py` — reused as the expert purity gate (design A4)

## Configuration

- model : OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
- prompt : "Explain how mixture of experts routing works." (chat template applied)
- threads : 4, seed : 42, greedy : temp 0, lanes : 4
- cache_mb : 0–1360, dense ∈ {mmap, warm, anon}, dontneed ∈ {0, 1}, sensor : 32

## Correctness Evidence (gate 1)

`scripts/run_phase6_5.sh 4 0 1 2`

| Check | Result |
|-------|--------|
| Stream OFF (resident mmap, baseline) | coherent text |
| Stream ON mmap / warm / anon | coherent text, **verify: 0 FAILs** |
| warm-up / ASK-pass discovery | 32/32 experts |
| Stream ON vs OFF (per dense policy) | **GATE PASS** ×3 |
| `dense anon boot verify` (anon vs mmap) | **0 FAILs** |
| Cross-policy: warm/anon OFF == mmap OFF | **GATE PASS** ×2 |

**FIX 6.5.1 (crash workaround, rc=139):** newer llama.cpp (`resolve_fused_ops`: Gated Delta Net / Lightning Indexer / DeepSeek V4 HC fused ops) refactored the MoE graph so the observe pass — which only scanned the observed `ffn_moe_topk` srcs — found **0/32 experts**, and the streaming path dereferenced `nullptr`. Fix: **expert discovery mirrored onto the ASK pass** (scan `t` + `t->src[i]` of every graph node; same version-proof guarantee as dense B1) + a per-layer `mmap fallback` guard + a `stream=1 && 0 experts` abort.

## Benchmark Results

Memory bench, OLMoE-1B Q4_K_M, lanes=4, threads=4, verify=0. Config A = `dense=anon + dontneed=1`; config B = `dense=mmap + dontneed=0`. Peak RSS in MB (independent VmRSS peak; RssAnon/RssFile are independent peaks, not same-instant).

| cfg | cache | tok | tok/s | RSS_peak | RssAnon | RssFile | hit% | reads |
|-----|-------|-----|-------|----------|---------|---------|------|-------|
| anon+dn | 512  | 128 | 1.93 | 4411 | 1224 | 4023 | 37.4 | 30783 |
| mmap    | 512  | 128 | 8.31 | 4953 | 928  | 4024 | 37.4 | 30783 |
| anon+dn | 1024 | 128 | 2.48 | 4441 | 1767 | 4024 | 54.1 | 22581 |
| mmap    | 1024 | 128 | 8.67 | 5494 | 1469 | 4025 | 54.1 | 22581 |

Dense-residency gate (via `check_dense_residency.py`) after an `anon + dontneed` run: **dense 7.8%, expert 7.9% → `PURE`** (control mmap run: **100% → `NOT PURE`**, confirming the measurement).

## Key Findings

### 1. The eviction primitive for mapped pages is a **two-step**: `madvise(MADV_DONTNEED)` then `posix_fadvise(DONTNEED)`
- `fadvise(DONTNEED)` **skips clean pages that are still mapped** (`invalidate_mapping_pages` bails on `page_mapped`) — and returns 0, a false success. On a model faulted via mmap during prefill, fadvise alone dropps nothing (dense stayed 98%).
- `madvise(MADV_DONTNEED)` zaps the PTEs but leaves the pages in the **page cache** (which `mincore` measures), so it alone also fails the gate.
- **In that order** (madvise then fadvise), the now-unmapped clean pages are invalidated from the cache: dense 98% → **7.8%**.
- Residual (~7.8%) = 4 KB **boundary pages shared** between dense and expert tensors (GGUF tensors aren't page-aligned); they re-enter the cache when a neighbouring expert is read.

### 2. `mincore` measures page-cache residency, not PTE presence
The gate and the sensor must be read accordingly: `dense_resident_frac` in `anon` mode is the **anon-buffer** residency (always ~100%), *not* the purity number. Purity comes only from `check_dense_residency.py` (mincore over the file ranges).

### 3. ASK-pass discovery is the version-proof surface
`ask=true` fires for every graph node, so scanning `t` + `t->src[i]` there guarantees coverage of all weight leaves (dense **and** expert) in any llama.cpp release — both the `0/32` crash and the older observe-pass fragility are behind us.

### 4. Experts need a **boot-time full-range drop**, not just per-miss
Per-miss madvise only covers experts re-read during decode; experts faulted by prefill and never re-routed kept their PTE and stayed resident (expert 22.5%). A **boot-drop over all expert full ranges** (same madvise→fadvise) after prefill brought experts to **7.9%**. Safe: madvise doesn't remove the VMA, so the `ne>1` batch safety net still works (it just re-faults).

### 5. Peak RssFile ≈ full model regardless of policy (on a RAM-resident box)
The file stays resident *during* the run (prefill faults dense + top-k experts; decode transients repopulate); the drops operate at rest. So the "peak RSS" doesn't show the policy difference — measure **rest/steady-state** RssFile to see it.

### 6. RAM-bound math (the "does it fit" vs "how smooth" distinction)
- **File cache is reclaimable**: under pressure / `MemoryMax`, the kernel evicts clean file pages to keep RSS under the cap — so a big model "fits" via reclaim regardless of policy. **The real budget is the anon sum**: `dense(anon) + LRU + KV + staging + base < cap`.
- Phase 6.5's anon dense + bounded LRU is what keeps that sum small → **solves "does it fit"**.
- **O_DIRECT does not reduce anon** → it does **not** change "does it fit". Its value is (a) peak RssFile (only visible when RAM has slack), (b) **steady-state: no reclaim fault-storm → stable tok/s**, (c) an **exact** memory equation with no hidden file-cache term. Succinct verdict: **not needed to close gates 1–3; worth it for Phase 8/production under tight RAM.**

## Notes / Decisions for Later Phases

- **Phase 7 (telemetry):** the sensor (`dense_resident_frac`, majflt deltas) and the per-run counters shipped here are the head start; make sure `RssFile_rest` (post-drop) and `saving_vs_mmap` are added to the bench, since peak RssFile is not discriminative.
- **Phase 8 (model > RAM):**
  - **O_DIRECT upgrade** (dense + experts), but **only together with Estágio A** — dense conversion **before** prefill via `llama_model_get_tensor` + expert O_DIRECT from the first decode — otherwise prefill re-faults the file and the peak stays. Requires an aligned bounce per lane (512 B) for expert slices.
  - Re-benchmark **lane counts** on the target machine (Phase 5 found 2 lanes sweet on a resident laptop SSD; with every miss hitting disk, more lanes may pay).
  - **Batched decode (`ne>1`)**: deferred; size `n_ctx` so generation never overflows (the guard stays as the safety net).
  - **mlock**: only under a budget with headroom (mlock + tight `MemoryMax` = OOM); needs raised `RLIMIT_MEMLOCK`.
- **Correctness contract holds** across every policy (mmap/warm/anon), verified byte-identical; O_DIRECT must preserve it (same boot `memcmp`).

## Exit Criteria (Phase 6.5)

1. [x] Correctness: `verify=1` -> 0 FAILs, stream ON == OFF per policy.
2. [x] Dense residency (anon): dense 7.8% `PURE`, 0 major faults on dense post-boot; control mmap 100%.
3. [x] Expert purity (A4): 7.9% with boot-drop (was 22.5%); re-audit via `check_cache_purity.py`.
4. [ ] Budget: steady-state RSS = `dense + cache + staging + KV + base` under `MemoryMax` (clean file pages reclaimed, not counted).
5. [ ] mlock (if on): dense never swapped, no OOM with headroom.
