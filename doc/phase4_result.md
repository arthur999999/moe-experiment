# Phase 4 — Expert LRU Cache + Strategy B (RESULTS)

**Status:** ✅ COMPLETE — correctness maintained, locality measured, memory cost quantified

## Summary

Phase 4 adds an LRU cache of experts in RAM plus Strategy B (compact expert buffers with
router-id remapping) on top of the Phase 3 streamer. Instead of reading all routed experts
from disk every token, recently-used experts are kept in a RAM cache; misses are the only
reads that hit the disk. Strategy B keeps only the n_expert_used (8) experts in compact
buffers instead of full-size fused tensors, substantially cutting the streaming's anonymous
memory footprint.

**Correctness gate: PASSED** — output is token-for-token identical with and without the cache.

## Files

- `src/cxx/phase4_stream.cpp` — streamer with LRU cache + Strategy B
- `scripts/build_phase4.sh` — build against compiled llama.cpp
- `scripts/bench_phase4.sh` — per-cache-size benchmark (tok/s, RssAnon peak, RssFile peak, hit rate, disk reads)

## Configuration

- model : OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
- prompt : "Explain how mixture of experts routing works." (chat template applied)
- n_predict : 128
- threads : 4
- seed : 42
- greedy : temp 0
- repack : OFF
- cache_mb : 0 to 2048 (0 = no cache / Phase 3 behavior)

## Benchmark Results (real hardware, model fits in RAM)

| cache | tok/s | RssAnon peak | RssFile peak | hit rate | disk_reads |
|-------|-------|--------------|--------------|----------|-----------|
| 0 MB | 10.18 | 125,368 kB | 4,121,868 kB | — | 48,768 |
| 64 MB | 6.86 | 200,396 kB | 4,121,560 kB | 0.0% | 48,768 |
| 256 MB | 7.15 | 397,776 kB | 4,121,900 kB | 0.0% | 48,768 |
| 512 MB | 10.64 | 677,788 kB | 4,121,884 kB | 92.9% | 3,486 |
| 1,024 MB | 10.76 | 1,181,916 kB | 4,122,040 kB | 97.5% | 1,206 |
| 2,048 MB | 10.71 | 1,510,500 kB | 4,121,664 kB | 97.7% | 1,116 |

RssAnon = anonymous RAM paid by the streaming driver (cache + compact buffers + KV + heap).
RssFile = file-backed RAM (the mmap'd model pages) — only relevant while the model fits.

## Key Findings

### 1. Correctness maintained

Output with the LRU cache is token-for-token identical to output without it.
The cache changes where expert bytes come from (RAM vs disk), never the math.

### 2. Cache cost is flat and predictable

Each MiB of cache costs roughly 1 MiB of RssAnon:

    RssAnon(cache=1024) - RssAnon(cache=0) ≈ 1.06 GB   (matches cache_mb)

Streaming base cost (no cache) ≈ 125 MB anon — the Strategy B compact buffers are cheap.

### 3. There is a hard threshold between 256 MB and 512 MB of cache

- ≤256 MB → **0% hit rate** (cache too small; LRU evicts everything)
- 512 MB → **92.9%** hit rate (48,768 → 3,486 disk reads)

This is the "cache below working set is useless" cliff — critical for sizing on devices.

### 4. Saturation at ~1.0–1.3 GB

- 1,024 MB → 97.5% hits; 2,048 MB → 97.7% (marginal gain)
- The active working set is ~1.0–1.3 GB for this model/prompt. Beyond that, RAM buys almost nothing.

### 5. tok/s is confounded on this machine — do not trust it yet

tok/s shows no clear pattern (6.9 to 10.8) because the model fits in RAM: all preads hit the
OS page cache, so cache hits save no real disk I/O. Meaningful tok/s comparisons require the
Phase 8 scenario (model > RAM) where a miss is real I/O.

### 6. O_DIRECT is not viable per-expert on this GGUF (tested, removed)

Verified empirically: opening with O_DIRECT succeeds, but the first pread() of an expert
fails with EINVAL because GGUF aligns tensors to 32 bytes, not 512/4096:

    [odirect] EINVAL at offset 176919360; falling back to buffered+DONTNEED

Decision: use plain buffered pread for now. If O_DIRECT is needed in Phase 8 (model > RAM),
either read aligned slabs covering experts or regenerate the GGUF with --align 4096.

### 7. Streaming survives real memory pressure

With systemd-run --scope -p MemoryMax=3G (model ~4.2 GB on disk), the streamed run completes
normally (11.24 tok/s, 97.5% hits) without OOM — evidence the design holds under pressure.

## Correctness Evidence

| Check | Result |
|-------|--------|
| Output cache=0 vs cache=1024 | ✅ identical (diff empty) |
| warm-up discovery | 32/32 experts |
| hit rate with 1 GB cache | 97.5% |
| disk reads 1 GB cache | 1,206 (vs 48,768 no cache) → 40× less I/O |

## Notes for Later Phases

- **Phase 5 (parallel I/O):** cache gives the locality stats; parallel lanes will reduce miss latency.
- **Phase 8 (model > RAM):** the investment that pays off is the cache + strategy B design;
  O_DIRECT should be revisited with aligned slabs or --align 4096 GGUFs.
- **Memory sizing rule measured:** total_stream_RAM ≈ 125 MB + cache_mb.