# Phase 6.5 — Memory Policy Design

**Status:** DESIGN — implementation pending
**Scope:** deterministic control of RAM on both sides of the model — **dense** (pinned resident) and **experts** (bounded budget) — so that the process footprint is *exact and measurable* instead of depending on the kernel's lazy page-cache behavior.
**Correctness contract:** unchanged — streamed output must remain byte-identical to the resident baseline under every memory policy.

---

## 1. Why this phase exists

Previous phases established three mechanisms that *overlap but do not substitute*:

| Mechanism | What it does | Nature |
|---|---|---|
| `posix_fadvise(DONTNEED)` | Drops **file-backed page-cache** pages for a byte range | precise, active, per-range |
| `mlock()` | Pins **anonymous** pages in RAM — non-reclaimable, non-swappable | absolute, per-buffer |
| `MemoryMax` (cgroup v2) | Global RSS cap; kernel reclaims **coldest clean file pages first**, then swaps anon | coarse, reactive, global |

The design goal is to use each where it is strong:

- **Dense weights** → *fix* in RAM: `anon` buffers + rebind (+ optional `mlock`).
- **Experts** → *bound* in RAM: LRU budget + durable `DONTNEED` (page cache must not hold a second copy).

### 1.1 The problem this phase fixes (prior findings)

Phase 4/5 proved `DONTNEED` works mechanically (mincore A/B, `check_cache_purity.py` verdict `PURE`), but also measured that its **effect is undone** — the mmap re-faults the same bytes through access paths outside the streaming callback:

1. **`verify`** — the `memcmp` against the mmap'd `->original` on every pread (runs in every benchmark row that reports `0 FAILs`);
2. **`restore_all()`** — resets expert `->data` to mmap originals between single-token decodes;
3. **batched decode (`ne>1`)** — the `if (t->ne[1] != 1) return true;` guard skips streaming entirely, so llama.cpp reads experts via mmap and repopulates the page cache;
4. **prefill** — runs with streaming disabled (warm-up), all mmap, no DONTNEED (acceptable: once, before streaming).

On the resident test machine this double-read made `DONTNEED` **net negative** (each miss costs 2 I/Os: pread + mmap fault — see 1024-token benchmarks: `cache=2GB` beats `cache=0`). The value of `DONTNEED` only materializes when the model **does not fit in RAM** (Phase 8), where the second copy is pure waste.

The consequence for *purity*: as long as those four paths exist, expert file pages are re-populated, so the expert footprint is **not bounded** by `cache_mb + staging` alone.

---

## 2. Design A — Durable expert page-cache purity

Goal: make `DONTNEED` *stick* so the expert footprint equals exactly `staging + LRU budget`.

### A1. Split verify from benchmark runs

- **Correctness run** (gate, one-off): `verify=1` → every pread memcmp'd against mmap → must stay `0 FAILs`.
- **Benchmark runs**: `verify=0` — byte-check replaced by **residency checks** (mincore), which never touch the mmap.

### A2. Remove `restore_all()` from the streamed decode path

- After warm-up, expert tensors stay rebound to staging across single-token decodes (no per-token restore).
- Restore is performed **only** around non-streamed phases: prefill (first decode) and batched decodes (`ne>1`).
- Safety: the correctness gate (stream ON == OFF) arbitrates; a restore must also precede **any** batch decode to keep that path valid.

### A3. Batched decode (`ne>1`) — explicit policy decision

Batched decodes (context shift) currently bypass streaming and repopulate expert pages via mmap, undoing purity.

- **Decision for 6.5:** avoid them — size `n_ctx` to the workload so generation never overflows (documented; `n_ctx` was already flagged as Phase-8 tuning).
- **Deferred (Phase 8):** real batched streaming (stride-aware top-k gather), mirroring BigMoeOnEdge's `gathered_.` handling.

### A4. Purity gate (mincore)

After any streamed run with `dontneed=1` (verify off, no-restore, no batch): `scripts/check_cache_purity.py` must report `verdict: PURE`.

New run metric: `expert_purity%` + `RssFile` steady-state at end-of-run.

---

## 3. Design B — Dense weights policy

### B1. Discovery (warm-up)

Capture dense tensors = **weight leaves** (`ggml_tensor` with `op == GGML_OP_NONE`, named, with a GGUF tensor match) **excluding** `*_exps` expert tensors. KV-cache tensors and graph inputs share the leaf shape and are filtered out by intersecting with the GGUF tensor set (same approach as BigMoeOnEdge's capture pass).

### B2. Byte ranges

`dense_ranges = byte_ranges(expert_ranges, file_size)` — the complement of the expert spans in the file, computed once (`rg_data_offset + tensor_offset` per dense tensor).

### B3. Policies (`--dense {mmap|warm|anon}`)

| Policy | Behavior | When |
|---|---|---|
| `mmap` | current behavior (file-backed, fault-storm when model > RAM) | resident-only testing |
| `warm` | `posix_fadvise(WILLNEED)` once; never DONTNEED that range | dense << RAM with margin |
| `anon` | **default for Phase 8**: read once into anon buffers, rebind, DONTNEED the file ranges | model > RAM |

### B4. `anon` pipeline (boot-time, once)

1. Allocate anon buffers (one per dense tensor; or one contiguous region + per-tensor slices).
2. `pread` in parallel through the existing `IoPool` lanes (dense set ≈ 2–4 GB → <1 s on NVMe).
3. `memcmp` against mmap **once** (boot verify), then:
4. Rebind each dense tensor's `->data` to its anon buffer; save `original`.
5. `fadvise(DONTNEED)` on the **dense file ranges** — clean by construction, because after the rebind **no compute path touches the dense mmap** (unlike experts, there is no double-I/O risk here).
6. Teardown: restore originals before `llama_free`.

### B5. Optional `mlock`

- `mlock()` each dense anon buffer (or `mlockall` scoped to the buffers).
- Requires `CAP_IPC_LOCK` / raised `RLIMIT_MEMLOCK` (`ulimit -l`, or `LimitMEMLOCK=infinity` in the systemd unit).
- **Hard rule:** only enable when the budget has headroom — `mlock` + tight `MemoryMax` converts graceful degradation into an OOM kill (pinned pages cannot be reclaimed).

### B6. Residency sensor

Sample every N tokens:
- `dense_resident_frac` — fraction of dense bytes physically resident (mincore over the anon buffers / smaps);
- `major faults` delta since last sample.

Purpose: detect the day any layer above stops being true (a stray non-rebound path touching the mmap, swap entering, unexpected reclaim). Reported in every metrics file; never silently zero.

---

## 4. Budget equation (validates the limit)

```
cache(LRU) = MemoryMax − dense(anon) − KV − staging − overhead
```

Expected per target (8 GB laptop):

| Component | Qwen3-30B (18.5 GB) | 27B-class (17 GB) |
|---|---|---|
| dense anon | ~2 GB | ~3–4 GB |
| staging (strategy A, 1 layer) | ~0.3 GB | ~0.3 GB |
| KV cache | ~1 GB | ~1 GB |
| overhead/base | ~0.6 GB | ~0.6 GB |
| **LRU cache headroom** | **~3.5–4 GB** | **~2–3 GB** |

Phase-4 locality: 512 MB→37% hit, 1 GB→53%, 2 GB→82% — the headroom above lands on the plateau where the cache is decisive.

---

## 5. Threading / lifetime invariants

- Dense boot conversion happens **before** the first streamed decode; lanes never mutate dense buffers after boot.
- Expert staging/lanes: unchanged from Phase 5 (`IoPool` jobs, main-thread LRU).
- Restore points: prefill + every batched decode + teardown.
- `mlock`/sensor run on the main thread; sensor is throttled so its cost lands in `mgmt` accounting, never hidden in compute.

---

## 6. Benchmark protocol

Harness: `bench_phase6_5.sh` (extends `bench_phase5.sh`).

Matrix: `dense ∈ {mmap, warm, anon}` × `mlock ∈ {0, 1}` × `cache ∈ {0, 512M, 1G, 2G}` × `tokens ∈ {128, 1024}` × `dontneed ∈ {0, 1}`.

Per-run metrics: `tok/s · RssAnon peak · RssFile peak · dense_resident_frac · major faults/token · expert_purity% · cache hit · disk_reads`.

---

## 7. Gates

1. **Correctness:** `verify=1`, `dontneed=1` run → `0 FAILs`, stream ON == OFF, per dense policy.
2. **Purity:** `check_cache_purity.py` → `PURE` after a ≥1024-token streamed run (dontneed stuck).
3. **Dense residency:** with `dense=anon` — post-boot, dense bytes NOT resident in page cache (mincore over file ranges), and zero major faults on dense after warm-up.
4. **Budget:** under `systemd-run -p MemoryMax=`, steady-state RSS tracks `dense + cache + staging + KV + base` (no unexplained file pages).
5. **mlock (if on):** dense never swapped; no OOM with headroom.

---

## 8. Deliverables

| File | Purpose |
|---|---|
| `src/cxx/phase6_dense.cpp` | discovery + ranges + policies + rebind/restore + DONTNEED + mlock + sensor |
| `scripts/build_phase6_5.sh` | build against compiled llama.cpp |
| `scripts/bench_phase6_5.sh` | benchmark harness (matrix above, `systemd-run` budget mode) |
| `scripts/check_dense_residency.py` | sensor analysis + report |
| `scripts/check_cache_purity.py` | reused as the expert purity gate |
| `doc/phase6_5_result.md` | results + decision record (populated with measured runs) |

---

## 9. Exit criteria

1. `anon`: dense out of page cache, compute only via anon, **0 major faults** on dense post-warm-up.
2. Experts: ≥1024-token run with `dontneed=1` + verify off + no-restore → `PURE` at end (durable).
3. Byte-identical per policy (`mmap`/`warm`/`anon`): `0 FAILs`.
4. Under `MemoryMax`: RSS = the exact equation in §4.
5. `mlock` (if enabled): dense resident, no OOM.

---

## 10. Risks / open decisions

- **No-restore correctness** — only safe with prefill/batch coverage; the gate arbitrates.
- **Batch `ne>1`** — deferred to Phase 8 (batched streaming); 6.5 avoids it via `n_ctx` sizing.
- **`warm`** — a kernel promise; only for dense ≪ RAM with margin.
- **`mlock` + tight `MemoryMax` = OOM** — enable only with headroom.
- **Dense discovery filtering** — weight-leaf vs KV vs graph input: intersect with GGUF tensor set.

---

## 11. Relationship to other phases

- **Phase 6 (overlap seam):** orthogonal — 6.5 changes *where bytes live*, 6 changes *when bytes arrive*. Both are prerequisites for Phase 8; either order works.
- **Phase 7 (telemetry):** 6.5 ships the first real per-run counters (`dense_resident_frac`, major faults, `expert_purity%`) — a head start.
- **Phase 8 (scaling):** this phase exists so Qwen3-30B / 27B-class actually *run* on 8 GB before any further expert tuning matters.

---

*References: prior findings in `doc/phase4_result.md` (DONTNEED mechanics, Strategy A, memory pressure), `doc/phase5_result.md` (IoPool, double-I/O at 1024 tokens), `scripts/check_cache_purity.py` (mincore A/B), and BigMoeOnEdge `core/src/moe/expert_stream_source.cpp` (dense-weights anon, vm_reserve/lazy commit, residency sensor) and `core/src/moe/router_hook.cpp` (leaf capture + gguf filtering).*

---