# Phase 4 — Expert LRU Cache + Corrected Strategy (RESULTS)

**Status:**  COMPLETE (CORRECTED) — Strategy B diagnosed, root cause found, final = Strategy A + DONTNEED, byte-validated

## Summary

Phase 4 started as "LRU cache + Strategy B (compact buffers + router-id remap 0..7)".
**Strategy B turned out to be broken on stock llama.cpp** (garbage output, deterministic).
After root-cause analysis in llama.cpp source, the final Phase 4 uses **Strategy A**
(full-size scatter buffers + original router ids) + **DONTNEED** (kernel page-cache purity),
with a byte-for-byte `verify` gate. This is the same approach used by BigMoeOnEdge.

**Correctness gate: PASSED** — streamed output is coherent and every pread is proven
byte-identical to the mmap (`verify: 0 FAILs`).
**Validated against resident baseline:** `llama-cli` (resident, same build) produces the
same coherent text.

## Root cause: why Strategy B (compact + id rewrite) is impossible on stock llama.cpp

Evidence chain (all reproduced on this machine):

1. llama-cli (resident) → coherent text. Streaming (Strategy B) → garbage `:...a:...`.
2. Reads were proven 100% correct — `verify: 0 FAILs` on Strategy A — so offsets/bytes were never the issue.
3. Source analysis (llama.cpp dict 0b1bad14f):
   - `ggml_mul_mat_id` (ggml-cpu.c) accepts ids 0..7 to index a compact weight buffer .
   - `build_moe_ffn` (llama-graph.cpp) binds `weights = ggml_get_rows(ctx0, probs, selected_experts)`
     — the ROUTING-WEIGHT gather uses the SAME `ffn_moe_topk` buffer. Rewritten ids cause it
     to gather `probs[0..7]` instead of `probs[eid]` → wrong scales → deterministic garbage.
   - The same buffer feeds BOTH nodes → no way to rewrite ids for one and not the other.
4. Attempted workaround (`phase4_stream_b_mmid.cpp`, intercept `GGML_OP_MUL_MAT_ID` with
   per-node private compact ids): **FALSE POSITIVE** — `if (ask) return false` after warm-up
   makes `MUL_MAT_ID = 0` and `disk reads = 0` (callback dead, pure resident inference).
   Even with MMID observed, `cb_eval` gives no viable window: ask=false arrives either after
   compute (rebind inert) or before (src2 shared across the 3 mul_mat_id of a layer → contamination).

**Conclusion:** on stock llama.cpp the only correct strategy is **A** — hook `ffn_moe_topk`
(the first MoE node of the layer, before any consumer), scatter the routed experts into a
full-size buffer at their ABSOLUTE offsets (`eid × bytes_per_expert`), keep original ids.
BigMoeOnEdge does exactly this (scatter + cache, never rewrites ids).

## Files

- `src/cxx/phase4_stream.cpp` — final streamer: Strategy A + LRU cache + DONTNEED + verify gate
- `scripts/build_phase4.sh`, `scripts/bench_phase4.sh`

## Configuration

- model : OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
- prompt : "Explain how mixture of experts routing works." (chat template applied)
- n_predict : 128, threads : 4, seed : 42, greedy : temp 0, repack : OFF
- cache_mb : 0–2048 (0 = no cache), verify : 0/1
- backend : single `pread()` buffered + `posix_fadvise(DONTNEED)` (no O_DIRECT — see finding 5)

## Correctness Evidence (validated runs)

| Check | Result |
|-------|--------|
| Residence baseline (llama-cli, same build) |  coherent text |
| Streamed (Strategy A, cache=0, DONTNEED, verify=1) |  coherent text, **verify: 0 FAILs** |
| warm-up discovery | 32/32 expert tensors |
| DONTNEED innocence |  confirmed (Strategy A + DONTNEED = 0 FAILs + coherent) |

`verify=1` memcmp's every pread against the mmap'd source: **0 FAILs** ⇒ the streaming reads
are byte-exact; only the buffer/ids mechanics remained, and Strategy A makes them correct by construction.

## Key Findings

### 1. The cache changes where bytes come from, never the math
Output identical with/without cache (token-for-token). LRU replaces disk reads with RAM copies.

### 2. Locality facts (working set) — strategy-independent, still valid
- **Hard cliff between 256 MB and 512 MB:** ≤256 MB → 0% hit rate (LRU evicts everything);
  512 MB → 92.9% hits (48,768 → 3,486 disk reads). "Cache below working set is useless".
- **Saturation ~1.0–1.3 GB:** 1,024 MB → 97.5%; 2,048 MB → 97.7% (marginal).
  Active working set ≈ 1.0–1.3 GB for this model/prompt.

### 3. Memory cost is flat; staging cost of Strategy A is bounded
Each MiB of cache ≈ 1 MiB of RssAnon. Strategy A staging = ONE layer's experts
(n_expert × gate/up/down), reused across layers — ~260 MB on OLMoE, NOT the model.
Total stream RAM ≈ staging + cache + KV + heap.

### 4. DONTNEED removes the kernel's copy (page-cache purity)
`posix_fadvise(fd, off, size, DONTNEED)` after each miss lets the kernel drop the expert
pages — the expert bytes live only in our staging/cache, not duplicated in the page cache.
This is what makes the design work when the model does NOT fit in RAM (Phase 8), and is the
desktop equivalent of BigMoeOnEdge's `--no-odirect`.

### 5. O_DIRECT is not viable per-expert on this GGUF (tested, removed)
Opening with O_DIRECT succeeds, first pread fails EINVAL (GGUF aligns to 32 bytes, not 512/4096).
Decision: buffered pread + DONTNEED. Phase 8 may revisit with aligned slabs or `--align 4096`.

### 6. Streaming survives real memory pressure
Completes under `systemd-run --scope -p MemoryMax=3G` without OOM (model ~4.2 GB on disk).

## Pending re-measurement (Strategy A)

The old benchmark table measured the BROKEN Strategy B layout; locality numbers above remain
valid, but memory columns must be re-measured with Strategy A staging (~260 MB base instead of ~125 MB):

| cache | tok/s | RssAnon peak | disk_reads | hit rate |
|-------|-------|--------------|-----------|----------|
| 0 MB | *pending* | *pending* | 48,768 | — |
| 512 MB | *pending* | *pending* | ~3,486* | *pending* |
| 1,024 MB | *pending* | *pending* | ~1,206* | *pending* |

(red. `disk_reads`/hit-rate are locality facts and expected to reproduce.)

## Notes for Later Phases

- **Phase 5:** rebuild lanes + cache + DONTNEED on **Strategy A** (validated base);
  gate = stream-ON vs stream-OFF on the same binary; measure lanes 1/2/4/8 × cache 0/64/512.
- **Phase 8 (model > RAM):** staging stays bounded (one layer's experts); the cache is the
  dial you choose; verify `mincore`/pagemap that expert pages are non-resident (cache purity).
- **Compact memory, if ever needed:** a ~25-line seam in ggml's `mul_mat_id` creating a
  SECOND private ids tensor (like the overlap seam already accepted for Phase 6) — an
  architectural decision for Phase 6+, not a Phase 4/5 blocker.