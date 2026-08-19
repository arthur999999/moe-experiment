# MoE Experiment — Expert Streaming Research

> **Research project:** running Mixture-of-Experts models far larger than available RAM, on CPU only, by streaming routed experts from disk.

This project explores **expert streaming** for MoE (Mixture-of-Experts) models: instead of loading every expert into RAM, only the experts a token actually routes to are read from disk, on demand, at the moment they're needed. The long-term goal is to run a **~27B-parameter MoE model on a laptop with 8GB of RAM and no GPU**, with output identical to running the model fully resident in memory.

This is inspired by and takes engineering ideas from prior work such as AirLLM, Apple's *LLM in a flash*, FlexGen, PowerInfer, EdgeMoE, and projects like [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge), which demonstrated this approach on Android phones using stock llama.cpp.

## Goal

Run a MoE model roughly the size of **Qwen3-30B-A3B** or **Gemma-27B-class** — well beyond an 8GB RAM budget — on a CPU-only laptop, with:
- **Lossless output**: byte-for-byte identical to the model running fully in RAM
- **No modified llama.cpp internals**: streaming implemented through its public evaluation hooks, upstream stays untouched
- **Measured, not assumed, performance**: every claim backed by a reproducible benchmark

## Background: why MoE makes this possible

A Mixture-of-Experts model is built from many small "expert" sub-networks, but each token only activates a handful of them (e.g. 2 out of 8, or 8 out of 64) via a small router network. This means:
- **Total parameter count** can be huge (tens of billions)
- **Active computation per token** is a small fraction of that
- **Memory challenge**: even though only a few experts are used per token, all experts must be *reachable*, because routing decisions change token to token

The core idea of this project: since only a few experts are touched per token, we never need to hold every expert in RAM at once. We can keep the small always-used parts of the model resident, and fetch just the routed experts from flash storage exactly when they're needed.

## Roadmap — Phases

The project is built incrementally. Each phase must pass its correctness gate before the next one begins.

### Phase 1 — Baseline (no streaming)
Run a small MoE model (**OLMoE-1B-7B**, Q4_K_M) fully resident in RAM using stock llama.cpp, with fixed prompt, seed, and thread count.
- Establish baseline tokens/sec and peak RSS memory
- Record a **reference output** (exact generated tokens under greedy decoding) — this becomes the correctness gate for every phase that follows
- Inspect the GGUF file's internal tensor layout to understand how experts are stored (fused tensor shape, quantization type, byte offsets per expert)

**Utilities:**
- `scripts/inspect_gguf.py` — Python script using gguf-py to inspect tensor shapes, offsets, and quantization types
- `scripts/compare_output.py` — Validates current llama.cpp output against the saved reference output

**Exit criteria:** reproducible baseline numbers, a saved reference output, and a documented map of the expert tensor layout.

### Phase 2 — Manual expert reads
Write a standalone reader that, given a (layer, expert index) pair, computes the correct byte offset inside the GGUF file and reads that slice directly from disk — without going through llama.cpp yet.
- Verify each manually-read expert slice matches, byte-for-byte, the same slice already loaded by mmap
- No inference happens in this phase — this is purely about proving the offset math is correct

**Script:** `scripts/phase2_manual_expert_read.py` — Tests 36 combinations (3 layers × 3 tensor kinds × 4 experts) to verify offset calculations. Also discovered that quantization types vary per-layer (not just per-tensor-kind), which is crucial for Phase 3.

**Exit criteria:** manual reads match mmap-loaded data exactly, across multiple layers and experts.

### Phase 3 — Naive synchronous streaming — COMPLETE

Hook llama.cpp's evaluation callback to intercept the router's top-k expert selection before the expert matmul runs, and substitute a just-in-time disk read for each selected expert instead of relying on mmap.
- No caching, no parallelism — one token, one layer, one expert read at a time
- Compare generated output against the Phase 1 reference output — must match exactly

**Status:** COMPLETE — the correctness gate **PASSED**. Streamed inference produces token-for-token identical output to the resident baseline (mmap), and matches the freshly regenerated reference output. Speed is poor at this stage, as expected — correctness came first.

**How it works:** the streamer hooks llama.cpp's public `cb_eval` (a `ggml_backend_sched_eval_callback`). During warm-up it discovers the expert tensors by scanning each node's `src[i]` for `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`. After warm-up it only observes the router nodes (`ffn_moe_topk-%d`, layer parsed via `sscanf`), reads that token's top-8 expert ids from the router buffer, `pread()`s exactly those gate/up/down slices from disk into scratch buffers, and rebinds the expert tensors' `->data` before the matmul runs.

**Utilities:**
- `src/cxx/phase3_stream.cpp` — the streamer (main + callback + pread + rebind)
- `src/cxx/debug_graph.cpp` — graph debug / discovery helper
- `scripts/build_phase3.sh` — build against the already-compiled llama.cpp
- `scripts/run_phase3.sh` — runs with fixed config, separates stdout (text) / stderr (logs)
- `scripts/validate_phase3.py` — compares streamer output vs reference (whitespace-normalized)
- `doc/phase3_result.md` — full results, telemetry, and key findings

**Telemetry (validation run):** `16 layers, 64 experts, top-8`; `[DIAG] layer0.gate data_offset=169841472 (expected 169841472)`; `warm-up: 32/32 experts discovered`; `Phase 3: 48768 reads, 128 tokens` = `(128 - 1) decodes x 16 layers x 8 experts x 3 tensors`.

**Key findings (details in doc/phase3_result.md):**
1. `gguf_get_tensor_offset()` is **relative to the tensor data section**, not the file — you must add `gguf_get_data_offset(ctx)` to get the absolute offset, or every `pread()` diverges.
2. Returning `false` for `ask == true` batches non-router nodes; only router nodes are observed after warm-up.
3. Expert weights appear as `src` of the `mul_mat_id` ops, never as standalone graph nodes.
4. The router buffer is `[8, 1, 1, 1]` of `int32_t` expert ids; strategy A (full-size scratch buffer per tensor kind) was used, compact id-remap (strategy B) is deferred to Phase 4+.
5. The prompt must be formatted with the model's chat template (`tokenizer.chat_template` + `llama_chat_apply_template`) to match reference generation.
6. Controlled experiment: streaming ON == streaming OFF. Any divergence vs an external reference is a config/prompt difference, not a streaming bug.

**Exit criteria:** streamed inference on OLMoE-1B-7B produces output identical to the resident baseline. Speed is expected to be poor at this stage — correctness comes first.

### Phase 4 — Expert LRU Cache + Strategy A — COMPLETE

Add an LRU cache of experts in RAM plus **Strategy A** (full-size scatter buffers with original router ids) + `posix_fadvise(DONTNEED)` for kernel page-cache purity. Strategy B (compact buffers + router-id remap to 0..7) was proven **impossible on stock llama.cpp** — the same `ffn_moe_topk` buffer feeds both the weight gather and the routing-weight gather, so rewriting ids for one breaks the other.

**Status:** COMPLETE — correctness gate **PASSED** (byte-for-byte verify: 0 FAILs), root cause documented, strategy switched to A.

**Key findings (details in doc/phase4_result.md):**
1. Strategy B is impossible on stock llama.cpp — the `ffn_moe_topk` buffer is shared between `mul_mat_id` (weights) and `ggml_get_rows` (routing weights). Rewriting ids breaks both. The only correct approach is Strategy A: scatter into full-size buffers at absolute expert offsets, keep original ids.
2. Locality facts confirmed: hard cliff between 256 MB and 512 MB (≤256 MB → 0% hit rate; 512 MB → 37.2% hit rate). Saturation at ~1.0–1.3 GB.
3. Each MiB of cache ≈ 1 MiB of RssAnon. Strategy A staging = one layer's experts (~260 MB on OLMoE), reused across layers.
4. DONTNEED removes the kernel's page-cache copy — expert bytes live only in staging/cache, not duplicated in page cache. This is the desktop equivalent of BigMoeOnEdge's `--no-odirect`.
5. O_DIRECT confirmed inviable per-expert on this GGUF (EINVAL due to 32-byte alignment); buffered pread + DONTNEED used instead.
6. Streaming survives real memory pressure: completes under `MemoryMax=3G` without OOM.
7. tok/s is lower with cache ON than OFF at small token counts (128) because cache overhead (hash lookup, LRU eviction, full-size buffer memcpy) dominates. With longer generations (512+), cache hits amortize the overhead and cache ON becomes faster.

**Utilities:**
- `src/cxx/phase4_stream.cpp` — final streamer: Strategy A + LRU cache + DONTNEED + verify gate
- `scripts/build_phase4.sh` — build against the already-compiled llama.cpp
- `scripts/bench_phase4.sh` — per-cache-size benchmark (tok/s, RssAnon peak, RssFile peak, hit rate, disk reads)
- `doc/phase4_result.md` — full results, root cause analysis, and key findings

**Telemetry (1 GB cache run, 128 tokens):** `32/32 experts discovered`; `Cache: hits=... misses=... hit_rate=53.1% disk_reads=22869`; `Phase4-FIX: 128 tokens, ~21.4 s, ~5.98 tok/s`; `RssAnon peak: 1,475,364 kB`; `verify: 0 FAILs`.

### Phase 5 — Parallel I/O — COMPLETE

Add multiple concurrent read lanes so several experts can be fetched from disk at once instead of sequentially.
- Measure the effect of thread/lane count on throughput — more lanes isn't always better on a laptop SSD, this must be measured, not assumed

**Status:** COMPLETE — correctness gate **PASSED** (0 FAILs verify across all lane/cache combos), benchmarks measured.

**How it works:** an `IoPool` thread pool distributes expert-read miss tasks across N lanes using a static slice partition. Each lane owns a mutex/CV + queue; a shared completion counter signals the main thread when all misses are done. The LRU cache remains single-threaded (main thread only) — lanes never touch the cache map, avoiding races.

**Key findings (details in doc/phase5_result.md):**
1. **Sweet spot: 2 lanes** for most configurations. At short runs (128 tok) 2 lanes is +18% over serial; at longer runs (1024 tok, no cache) 8 lanes reaches +39%.
2. **8 lanes is often slower than 2–4** — context switching and completion-counter contention dominate when miss batches are small or cache reduces misses.
3. **With cache enabled,** parallel I/O helps little because few misses reach the pool.
4. Pool overhead (mutex, CV) is ~15% at 1 lane vs Phase 4 serial, but recovers at 2+ lanes.
5. All locality findings from Phase 4 confirmed unchanged (same cache algorithm).

**Utilities:**
- `src/cxx/phase5_stream.cpp` — streamer with IoPool (Strategy A + LRU + DONTNEED + parallel lanes)
- `scripts/build_phase5.sh` — build against compiled llama.cpp
- `scripts/bench_phase5.sh` — benchmark harness with optional RAM limiter (`-m <budget_mb>`)
- `scripts/run_phase5.sh` — self-contained gate: stream ON == stream OFF
- `scripts/check_cache_purity.py` — mincore-based tool to verify DONTNEED drops expert pages

**Telemetry (128 tok, cache=0, 2 lanes):** `16 layers, 64 experts, top-8, cache=0MiB, lanes=2`; `Phase5: 128 tokens, 17.09s, 7.49 tok/s`; `Cache: disabled. disk_reads=48768`; `verify: 0 FAILs`.

### Phase 6.5 — Dense-Weights Policy + Durable Page-Cache Purity — COMPLETE

Extend Phase 5 with a **dense (non-expert) weight policy** and make the page-cache eviction **durable** on both sides of the model — experts (bounded by LRU budget) and dense (pinned resident in anon buffers).

**Status:** COMPLETE — correctness gate green (6/6, stream ON == OFF per policy), dense-residency gate green (dense 7.8% `PURE`), memory bench delivered.

**Design (doc/phase6_5_design.md):**

Three mechanisms that overlap but do not substitute:

| Mechanism | What it does | Nature |
|---|---|---|
| `posix_fadvise(DONTNEED)` | Drops **file-backed page-cache** pages for a byte range | precise, active, per-range |
| `mlock()` | Pins **anonymous** pages in RAM | absolute, per-buffer |
| `MemoryMax` (cgroup v2) | Global RSS cap; kernel reclaims coldest clean file pages first | coarse, reactive, global |

**Design A — Durable expert page-cache purity:**
- A1. Split verify from benchmark runs (benchmark runs pass `verify=0`, memcmp replaced by mincore)
- A2. `restore_all()` removed from the single-token decode path (experts stay rebound across decodes)
- A3. Batched decode (`ne>1`) avoided via `n_ctx` sizing; guard stays as safety net
- A4. Purity gated by `check_cache_purity.py` (mincore)

**Design B — Dense (non-expert) weight policy:**
- B1. Discovery on the ASK pass (scans every graph node = version-proof)
- B2. Byte ranges: complement of expert spans in the file
- B3. Three policies: `mmap` (baseline, file-backed), `warm` (WILLNEED once, never DONTNEED'd), `anon` (pread into anon buffers, rebind, DONTNEED the file ranges)
- B4. Anon pipeline: page-aligned buffers, parallel pread via IoPool, boot memcmp verify, tensor rebind, fadvise(DONTNEED) + madvise(MADV_DONTNEED) on dense ranges
- B5. Optional `mlock` (best-effort, rlimit-aware)
- B6. Residency sensor (mincore over anon buffers or MAP_SHARED view, major-fault deltas)

**Key findings (details in doc/phase6_5_result.md):**
1. The eviction primitive for mapped pages is **two-step**: `madvise(MADV_DONTNEED)` then `posix_fadvise(DONTNEED)`. `fadvise` alone skips clean mapped pages (false success); madvise zaps PTEs, then fadvise invalidates cache. Dense: 98% → 7.8%.
2. `mincore` measures page-cache residency, not PTE presence — the sensor and the purity gate must be read accordingly.
3. ASK-pass discovery is the version-proof surface — scanning `t` + `t->src[i]` for every graph node guarantees coverage in any llama.cpp release (fixes the `0/32` crash from newer llama.cpp fused-op graphs).
4. Experts need a **boot-time full-range drop**, not just per-miss — per-miss madvise only covers re-read experts; a boot-drop over all expert ranges brought experts from 22.5% to 7.9%.
5. Peak RssFile ≈ full model regardless of policy on a RAM-resident box — measure rest/steady-state instead.
6. RAM-bound math: file cache is reclaimable under pressure, so "does it fit" is about the anon sum `dense(anon) + LRU + KV + staging + base < cap`.

**Utilities:**
- `doc/phase6_5_design.md` — design document for memory policy
- `doc/phase6_5_result.md` — full results and key findings
- `src/cxx/phase6_5_stream.cpp` — streamer with dense policy + boot-drop + residency sensor
- `scripts/build_phase6_5.sh` — build against compiled llama.cpp
- `scripts/run_phase6_5.sh` — self-contained correctness gate (stream ON == OFF per dense policy + cross-policy)
- `scripts/bench_phase6_5.sh` — memory-focused benchmark (peak RSS + RssAnon/RssFile split)
- `scripts/check_dense_residency.py` — gate 3: mincore over dense file ranges (PURE if < 15%)

**Correctness:** `scripts/run_phase6_5.sh 4 0 1 2` — GATE PASS ×6 (stream ON == OFF for mmap/warm/anon, cross-policy rebind lossless, `verify: 0 FAILs`, boot verify `0 FAILs`).

### Phase 6 — I/O / compute overlap
Overlap expert reads for the *next* computation with the *current* matmul, hiding disk latency behind compute time.
- Still must remain byte-identical to the reference output — overlap changes timing, not results

### Phase 7 — Telemetry
Instrument every run with a per-token breakdown: time spent in flash I/O, cache management, and compute, plus cache hit rate and bytes read per token. Mark anything that can't be measured directly as unmeasured rather than silently reporting zero.

### Phase 8 — Scaling up
Repeat Phases 3–7's validated pipeline against progressively larger models that exceed the 8GB budget by a larger margin:
1. **Qwen1.5-MoE-A2.7B** — small excess over budget, first real memory-pressure test
2. **Qwen3-30B-A3B** (~18.5GB, Q4_K_M) — several times over budget, dense-weight memory policy becomes important
3. **Target: a ~27B-class MoE model** (e.g. Gemma-MoE in the 26–27B range) running end-to-end on 8GB RAM, CPU only, with output validated against a resident-mode reference generated on a machine with enough RAM to run it normally

**Exit criteria for the project:** the ~27B model generates coherent output on an 8GB RAM / CPU-only laptop, at a measured (not estimated) tokens/sec, with a full telemetry breakdown of where the time goes.

## Key design principles carried through every phase

- **Correctness before speed.** Every phase re-validates against the frozen reference output from Phase 1. A performance win that breaks correctness is not a win.
- **Measure, don't assume.** Every number in this README under "Benchmarks" must come from an actual run on real hardware, not an estimate.
- **Build on stock llama.cpp.** No upstream modifications — streaming works through llama.cpp's public evaluation callback, so tracking new llama.cpp releases stays a routine submodule update.
- **Fixed threading for reproducibility.** Thread count is fixed across all benchmark runs, since floating-point reduction order (and therefore exact output) can shift with thread count.

## Installation

### Requirements
- Python 3.10+
- CMake, a C/C++ compiler (build-essential or equivalent)
- 8GB+ RAM (the whole point of the project is making this sufficient for models that would otherwise need much more)
- Enough free disk space for the model files in use (tens of GB for later phases)
- [huggingface-hub](https://huggingface.co/docs/huggingface_hub) Python library (for downloading models)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/moe-experiment.git
cd moe-experiment

# llama.cpp as a submodule (used as-is, unmodified)
git submodule update --init --recursive

# Python virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install huggingface-hub for model downloads
pip install huggingface-hub
```

### Download Phase 1 Model

This project uses **OLMoE-1B-7B-0924-Instruct** (Q4_K_M quantized) as the baseline model for Phase 1.

#### Option 1: Using the download script

```bash
# Make the script executable and run it
chmod +x download_model.sh
./download_model.sh
```

#### Option 2: Manual download (Python)

```bash
# Create models directory
mkdir -p models

# Download using Python/huggingface_hub
python << 'EOF'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="bartowski/OLMoE-1B-7B-0924-Instruct-GGUF",
    filename="OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
    local_dir="./models",
    local_dir_use_symlinks=False
)
EOF
```

The model file (~4.2GB) will be saved to `./models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf`.

### Building llama.cpp (baseline, Phase 1)

```bash
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j$(nproc)
```

### Phase 1 Utilities

#### scripts/inspect_gguf.py

Inspects the GGUF file to extract tensor metadata (shapes, offsets, quantization types):

```bash
python scripts/inspect_gguf.py
```

**Output:** Architecture metadata, expert tensor shapes, file offsets, and per-expert byte calculations.

#### scripts/compare_output.py

Validates that current llama.cpp runs match the saved reference output:

```bash
# Basic usage (uses default config)
python scripts/compare_output.py

# Pass extra flags to llama-cli
python scripts/compare_output.py --extra-args --some-flag
```

**Output:** "MATCH" if output identical to reference, "MISMATCH" with diff if different.

**Note:** The reference output is hardware/build-specific. See [doc/model_layout_notes.md](doc/model_layout_notes.md) for portability details.

## Benchmarks

Populated as each phase completes. No numbers are published here until they've been measured on real hardware with a fixed, documented configuration (device, thread count, prompt, seed).

All runs: OLMoE-1B-7B-0924-Instruct Q4_K_M, prompt *"Explain how mixture of experts routing works."*, 4 threads, seed 42, greedy (temp 0). Measured on a machine where the model fits in RAM.

### 128 tokens

At 128 tokens the cache overhead (hash lookup, LRU eviction, full-size memcpy) is not amortized — cache=0 is the fastest. Parallel I/O (lanes) helps only when misses are many.

| Phase | Streaming | tok/s | RssAnon peak | disk_reads | Hit rate | Output verified |
|-------|-----------|-------|-------------|-----------|----------|-----------------|
| 4 (cache=0) | Naive, no cache | 7.45 | 348,472 kB | 48,768 | — | PASS |
| 4 (cache=64 MB) | LRU + Strategy A | 4.59 | 423,496 kB | 48,768 | 0.0% | PASS |
| 4 (cache=512 MB) | LRU + Strategy A | 5.59 | 919,444 kB | 30,636 | 37.2% | PASS |
| 4 (cache=1 GB) | LRU + Strategy A | 5.98 | 1,475,364 kB | 22,869 | 53.1% | PASS |
| 4 (cache=2 GB) | LRU + Strategy A | 5.84 | 2,567,780 kB | 9,237 | 81.1% | PASS |
| 5 (cache=0, lanes=1) | Parallel I/O | 6.35 | — | 48,768 | — | PASS |
| 5 (cache=0, lanes=2) | Parallel I/O | **7.49** | — | 48,768 | — | PASS |
| 5 (cache=0, lanes=4) | Parallel I/O | 6.93 | — | 48,768 | — | PASS |
| 5 (cache=0, lanes=8) | Parallel I/O | 5.90 | — | 48,768 | — | PASS |
| 5 (cache=512 MB, lanes=1) | Parallel I/O | 4.50 | — | 30,636 | 37.2% | PASS |
| 5 (cache=512 MB, lanes=2) | Parallel I/O | 3.94 | — | 30,636 | 37.2% | PASS |
| 5 (cache=512 MB, lanes=4) | Parallel I/O | 4.61 | — | 30,636 | 37.2% | PASS |
| 5 (cache=512 MB, lanes=8) | Parallel I/O | 4.44 | — | 30,636 | 37.2% | PASS |

### 1024 tokens

At 1024 tokens, `cache=2 GB` overtakes `cache=0` (+12%). The reason: `posix_fadvise(DONTNEED)` in the no-cache path evicts expert pages from the page cache *after* `pread()`, but llama.cpp accesses the same bytes via mmap immediately after for the matmul — causing a **major page fault** (re-read from disk). With cache hits, no `pread` / no `fadvise` occurs, so pages stay in the page cache and the mmap finds them without fault. Each cache hit saves **two I/Os** (pread + mmap fault), which is enough to win when amortized over enough tokens.

| Phase | Streaming | tok/s | RssAnon peak | disk_reads | Hit rate | Output verified |
|-------|-----------|-------|-------------|-----------|----------|-----------------|
| 4 (cache=0) | Naive, no cache | 7.71 | 348,476 kB | 183,936 | — | PASS |
| 4 (cache=64 MB) | LRU + Strategy A | 5.21 | 424,024 kB | 183,936 | 0.0% | PASS |
| 4 (cache=1 GB) | LRU + Strategy A | 6.02 | 1,476,020 kB | 86,469 | 53.0% | PASS |
| 4 (cache=2 GB) | LRU + Strategy A | **8.64** | 2,567,776 kB | 32,343 | 82.4% | PASS |
| 5 (cache=0, lanes=1) | Parallel I/O | 5.89* | — | 183,936 | — | PASS |
| 5 (cache=0, lanes=2) | Parallel I/O | 6.53* | — | 183,936 | — | PASS |
| 5 (cache=0, lanes=8) | Parallel I/O | **8.21*** | — | 183,936 | — | PASS |

\* Phase 5 runs at 1024 n_pred generated 480 tokens (context limit of 512 reached; ~32 tokens used by prefill). Phase 4 ran at full 1024 — direct comparison is approximate.

### 3000 tokens

At 3000 tokens the benefit disappears: the context window (default 512) overflows, triggering expensive context shifting. Attention compute (O(n²)) dominates, and I/O becomes a negligible fraction. All configurations converge.

| Phase | Streaming | tok/s | RssAnon peak | disk_reads | Hit rate | Output verified |
|-------|-----------|-------|-------------|-----------|----------|-----------------|
| 4 (cache=0) | Naive, no cache | 6.75 | 348,476 kB | 183,936 | — | PASS |
| 4 (cache=64 MB) | LRU + Strategy A | 5.22 | 440,508 kB | 183,936 | 0.0% | PASS |
| 4 (cache=1 GB) | LRU + Strategy A | 6.55 | 1,505,576 kB | 86,469 | 53.0% | PASS |
| 4 (cache=2 GB) | LRU + Strategy A | 6.83 | 2,597,332 kB | 32,343 | 82.4% | PASS |

- **RssAnon** = anonymous RAM paid by the streaming driver (cache + staging buffers + KV + heap). The model itself (4.2 GB) lives in mmap'd file-backed pages (RssFile ~4.1 GB), not counted here.
- **disk_reads** = total `pread()` calls in the run. At 0% hit rate: 128 tokens → 48,768 reads (381 per generated token), 1024+ tokens → 183,936 (higher absolute but fewer per-token because batches >1 during context shifting don't trigger the callback).
- The **0% hit rate** at 64 MB confirms the working set is far larger — the LRU evicts every expert before re-use. Hit rate climbs with cache size: 53% at 1 GB → 82.4% at 2 GB.
- At **1024 tokens**, `cache=2 GB` is the fastest (8.64 tok/s, +12% over cache=0). At **128** and **3000** tokens, cache does not help — at 128 it is not amortized, at 3000 the context overflow overhead dominates.
- The cache is expected to matter most in **Phase 8** (model > RAM), where a miss means real page-fault I/O rather than a fast SSD read, and the context (n_ctx) will be tuned to match the workload.

## Development

```bash
# Run tests
pytest -v

# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## Resources

### Papers
- [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) — original MoE paper
- [Switch Transformers (2022)](https://arxiv.org/abs/2101.03961) — scaling MoE to trillion parameters
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) — Hugging Face overview

### Prior art on edge/CPU streaming
- [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) — expert streaming on Android via stock llama.cpp, direct inspiration for this project's approach
- [flash-moe](https://github.com/danveloper/flash-moe) — Metal-based expert streaming from SSD on Apple Silicon

### Models used across phases
- [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924) — Phase 1–3 baseline and streaming validation model
- [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) — Phase 8, first real memory-pressure test
- [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) — Phase 8, several-times-over-budget test
- Target ~27B-class MoE model — final Phase 8 milestone, exact model TBD based on GGUF availability and architecture support

### Tooling
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF inference engine used unmodified, streaming hooks into its public API
- [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) — GGUF format reading, used for tensor layout inspection

### Project Files

```
moe-experiment/
├── models/                          # Downloaded model files (gitignored)
│   └── OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
├── doc/                              # Documentation and reports
│   ├── reference_output.txt             # Phase 1 correctness gate (committed)
│   ├── reference_output_raw.txt         # Raw llama-cli output (committed)
│   ├── model_layout_notes.md            # Detailed model analysis
│   ├── phase3_result.md                 # Phase 3 results and key findings
│   ├── phase4_result.md                 # Phase 4 results and key findings
│   ├── phase5_result.md                 # Phase 5 results and key findings
│   ├── phase6_5_design.md               # Phase 6.5 design document
│   └── phase6_5_result.md               # Phase 6.5 results and key findings
├── scripts/                         # Build, run, and validation scripts
│   ├── build_phase3.sh
│   ├── build_phase4.sh
│   ├── build_phase5.sh
│   ├── build_phase6_5.sh
│   ├── run_phase3.sh
│   ├── run_phase5.sh
│   ├── run_phase6_5.sh
│   ├── bench_phase5.sh
│   ├── bench_phase6_5.sh
│   ├── check_cache_purity.py
│   ├── check_dense_residency.py
│   ├── compare_output.py                # Output validator
│   ├── inspect_gguf.py                  # GGUF file inspector
│   ├── phase2_manual_expert_read.py     # Manual expert offset verifier
│   └── validate_phase3.py               # Phase 3 output validator
├── src/cxx/                         # C++ streaming code (Phase 3+)
│   ├── phase3_stream.cpp
│   ├── phase4_stream.cpp
│   ├── phase5_stream.cpp
│   ├── phase6_5_stream.cpp
│   ├── debug_graph.cpp
│   └── ...
└── ...
```

**Important:** `doc/reference_output.txt` is committed as the correctness gate, but output is only reproducible on the exact hardware/llama.cpp build that generated it. See doc/model_layout_notes.md for details.

## Research notes

### Why GGUF
GGUF is the format this project relies on for everything: quantized weight storage, per-tensor metadata (including expert count and routing width), and — critically for this project — a layout where expert weights are addressable at known byte offsets, which is what makes selective disk reads possible in the first place.

### Known challenges going in
1. **Expert switching latency** — every streamed expert read adds latency the resident baseline doesn't pay
2. **Numerical determinism across thread counts** — must be pinned for the correctness gate to be meaningful
3. **Memory pressure behavior differs by OS** — how aggressively the OS reclaims pages under pressure affects which memory policy (mmap vs. pinned/anonymous) works best; this must be measured on the actual target laptop, not assumed from prior work on other platforms
4. **Router accuracy / cache locality** — how "hot" a small set of experts stays across tokens directly determines how effective caching can be, and this is model- and prompt-dependent

### Future directions (post-27B milestone)
- [ ] Expert merging/compression
- [ ] Speculative expert prefetching based on partial router signals
- [ ] Hierarchical expert organization
- [ ] Revisiting quantization choices per expert "temperature" (hot vs. cold experts)

## License

MIT License — see [LICENSE](LICENSE) for details.

This project includes third-party dependencies:
- llama.cpp — MIT License
- gguf-py — MIT License
- Model weights from Hugging Face — subject to their respective licenses

## Acknowledgments

- The [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) project, whose architecture and benchmark methodology this project's plan is directly modeled on
- Hugging Face Hub for model hosting
- The llama.cpp community for the GGUF format and inference engine
- The OLMoE, Qwen, and Gemma teams for open MoE model releases

---

**Note:** This is experimental research software under active development. Every benchmark number in this README reflects an actual measured run, or is explicitly marked `*pending*` until it is.