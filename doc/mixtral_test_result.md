# Mixtral 8x7B — Phase 6.5 Memory Test Result

**Status:** RESULT — in progress (measurement + next levers)
**Scope:** Phase 6.5 memory policy on Mixtral-8x7B Q4_K_M, CPU-only, NVMe.
**Device summary:** Mixtral 8x7B Q4_K_M; NVMe storage; CPU-only inference;
n_threads was 8 in the smoke test.

---

## 1. Executive summary

- **Dense policy: DONE and correct.** 227 tensors, 1081 MiB, `anon` mode,
  boot verify 0 FAILs, `resident_frac=100.0%`, major faults 49 total (~2.46/tok).
- **Dense is no longer the bottleneck.**
- **The remaining cost is the expert path.** `disk_reads=2508`, hit rate **34.3%**
  → ~65% of expert reads/misses are served from disk.
- **Measured result: 20 tokens in 171.28 s → 0.12 tok/s.**
- **The gap to BigMoeOnEdge's ~1 tok/s is caused by two missing levers**:
  cross-layer **overlap** and **O_DIRECT** — plus a model-level difference.

---

## 2. Measured run (this log)
```
warm-up: 64/64 experts
dense: mode=anon tensors=227 bytes=1081 MiB
dense anon boot verify: 0 FAILs
dense: anon ready — 227 tensors rebound, 1081 MiB total
Phase6_5: 20 tokens, 171.28s, 0.12 tok/s
Cache: hits=1332 misses=2508 hit_rate=34.3% disk_reads=2508
verify: 0 FAILs (expert preads == mmap)
Dense: mode=anon tensors=227 1081MiB resident_frac=100.0% majflt_total=49 (2.46/tok)
purity: expert_dontneed=0 dense_dontneed=0
```

Interpretation:

- **Dense is resident and healthy** (100%, 0 FAILs, low major faults).
- **Experts are still I/O-bound:** 2508 disk reads / 20 tokens ≈ **125 misses/token**,
  because the 64 MiB LRU cache here is far too small (it holds less than one expert
  span). This is the dominant cost.
- **`expert_dontneed=0`** — with DONTNEED off, expert reads leak into the page cache,
  so the model's file pages are getting read/written every token. That makes the
  I/O cost ~2× what a clean, bounded cache design would pay.

---

## 3. What this result proves

| Claim | Evidence |
|---|---|
| Dense anon residency works | 227 rebind, 100% resident, majflt only 2.46/tok |
| Expert stream is the hot path | 2508 disk_reads/20 tok ≈ 125 expert misses/tok |
| Cache (64 MiB) is far too small | hit_rate 34.3% vs the ~60–80% needed |
| Overlap + O_DIRECT are not wired yet | no overlap / O_DIRECT code present this run |

---

## 4. Why it is stuck at 0.12 tok/s

Two compounding costs:

1. **Small cache → expert thrash.** The 64 MiB LRU holds <1 expert span. Every token
   re-reads the routed experts from disk (2508 reads/20 tok). That is the
   I/O-bound floor.
2. **No overlap + no O_DIRECT.** Without cross-layer prefetch, a layer's disk read
   must fully finish before that layer computes. Without O_DIRECT, the expert path
   also runs through the page cache (page cache double-copy + pollution).

Then the system is still on the "cache + I/O refactoring" leg — the compute-bound
step (which would see CPU math as the only cost) has not been reached.

---

## 5. What to do in the real project (in priority order)

### 5.1 Fix the expert cache / locality (biggest win)
- Raise `cache_mb` meaningful: from 64 MB to **2–4 GiB or higher** so the hot
  expert working set is cached in RAM.
- Test sweep `cache ∈ {64M, 1G, 2G, 4G}` and pick the value where hit-rate goes
  clearly above ~60%. The result is: fewer disk reads per token.
- Add LRU eviction logging (hit rate vs reads) to confirm the working set is bounded.

### 5.2 Add cross-layer overlap (O/overlap)
- Right now `np.submit()` blocks until the lane is done (`done_cv.wait`). That makes
  the current-layer I/O synchronous. Overlap issues the *next* layer's reads while
  the current layer computes → flash latency hides under compute.
- Expected: I/O-bound → compute-bound. On NVMe, **~2–2.5×** from this alone.

### 5.3 Add O_DIRECT for expert reads
- Call the model file `fd` with `O_DIRECT` (or `madvise` + pread into aligned
  buffers) so the expert spans bypass the OS page cache entirely. This removes the
  double-copy and the page-cache pollution.
- Expected: **~1.2–1.5×** on top of overlap.

### 5.4 Make expert purity durable (`dontneed=1`)
- With the cache larger and overlap in, enable `dontneed=1`, verify off, no per-token
  restore → the streaming callback is the only access path, and the expert page-cache
  footprint becomes bounded by `staging + LRU`, not whole file.

### 5.5 Do not chase DeepSeek's 1 tok/s directly on Mixtral
- Mixtral Q4_K_M has `13B active per token` at Q4 (~6–7 GB of Q4 weights).
- DeepSeek V4 Flash (284B-A13B) also has ~13B active, but at **2-bit** (~2–3 GB).
- So per-token compute on Mixtral is **~2× heavier** — software alone can't reach an
  identical 1 tok/s. The goal for Mixtral is **compute-bound**, not DeepSeek's number.

---

## 6. Expected end-state (projection on this machine)

| Milestone | Target tok/s | Approx gain |
|---|---|---|
| Current (64M cache, no overlap/O_DIRECT) | 0.12–0.20 | 1× |
| + cache 2–4 GiB (hit >=60%) | 0.25–0.5 | ~2–3× |
| + cross-layer overlap | 0.4–0.7 | ~3–4× |
| + O_DIRECT + stable purity | 0.6–0.9 | ~4–6× |

These are realistic on NVMe + CPU. The 0.12 → ~0.5–0.9 path is the target.

---

## 7. Benchmark protocol (next run)

Sweep a small matrix:

- `dense ∈ {anon}`
- `cache ∈ {64M, 1G, 2G, 4G}`
- `overlap ∈ {0, 1}`
- `O_DIRECT ∈ {0, 1}`
- `dontneed ∈ {0, 1}`
- `tokens ≥ 128–1024` (long enough to warm the LRU and separate cold-start)
- `verify=0` for benchmark (verify enabled only in correctness gate)

Per-run metrics: `tok/s · hit_rate · disk_reads/token · donneed bytes · resident_frac`.

---

## 8. Known open items

- **Cache 64 MiB default is useless** for Mixtral (expert span > 64 MB). Must raise.
- **No overlap yet** — the biggest code lever.
- **No O_DIRECT yet** — read path still goes through page cache.
- **`dontneed=0` this run** — purity not yet measured here.

---

## 9. Reference

- [1] BigMoeOnEdge README + architecture doc: dense-weights anon, O_DIRECT, overlap,
  cache sizing, on-device measurements.
- [2] This Phase 6.5 design (same repo).
- [3] Deduction from this run's log.

---

## 10. Link to future phases

- **Phase 8** needs dense-anyone + overlap + O_DIRECT to run bigger-than-RAM models.
  This result is the reference state until those levers land.

---

### Update the file once you change the knobs

Re-measure with at least 128–1024 tokens (not 20) once you apply 5.1–5.3 so the
numbers reflect a steady-state working set, and re-verify byte-identity after
enabling overlap + O_DIRECT (your correctness gate).