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

### Phase 4 — Expert LRU Cache + Strategy B — COMPLETE

Add an LRU cache of experts in RAM plus Strategy B (compact expert buffers with router-id remapping) on top of the Phase 3 streamer. Instead of reading all routed experts from disk every token, recently-used experts are kept in a RAM cache; misses are the only reads that hit the disk. Strategy B keeps only the n_expert_used (8) experts in compact buffers instead of full-size fused tensors, substantially cutting the streaming's anonymous memory footprint.

**Status:** COMPLETE — correctness gate **PASSED**, memory cost quantified, cliff threshold measured.

**Key findings (details in doc/phase4_result.md):**
1. Output with the LRU cache is **token-for-token identical** to output without it — cache changes where bytes come from, never the math.
2. Each MiB of cache costs roughly 1 MiB of RssAnon — cost is flat and predictable.
3. **Hard threshold between 256 MB and 512 MB:** ≤256 MB cache → 0% hit rate (evicts everything); 512 MB → 92.9% hit rate (40× fewer disk reads).
4. Saturation at ~1.0–1.3 GB: 1,024 MB → 97.5%, 2,048 MB → 97.7% (marginal gain).
5. tok/s is confounded on this machine (model fits in RAM) — meaningful throughput comparison requires Phase 8 (model > RAM).
6. O_DIRECT confirmed inviable per-expert on this GGUF (EINVAL due to 32-byte alignment); plain buffered pread used.
7. Streaming survives real memory pressure: completes normally under `MemoryMax=3G` without OOM.

**Utilities:**
- `src/cxx/phase4_stream.cpp` — streamer with LRU cache + Strategy B
- `scripts/build_phase4.sh` — build against the already-compiled llama.cpp
- `scripts/bench_phase4.sh` — per-cache-size benchmark (tok/s, RssAnon peak, RssFile peak, hit rate, disk reads)
- `doc/phase4_result.md` — full results, telemetry, and key findings

**Telemetry (1 GB cache run):** `32/32 experts discovered`; `Cache: hits=... misses=... hit_rate=97.5% disk_reads=1206`; `Phase 4: 128 tokens, ~11.9 s, ~10.76 tok/s`; `RssAnon peak: 1,181,916 kB`.

### Phase 5 — Parallel I/O
Add multiple concurrent read lanes so several experts can be fetched from disk at once instead of sequentially.
- Measure the effect of thread/lane count on throughput — more lanes isn't always better on a laptop SSD, this must be measured, not assumed

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

| Phase | Model | RAM budget | Streaming | tok/s | Peak RSS | Output verified vs. reference |
|-------|-------|-----------|-----------|-------|----------|-------------------------------|
| 1 | OLMoE-1B-7B (Q4_K_M) | 8GB | No (baseline) | *pending* | *pending* | N/A (this **is** the reference) |
| 3 | OLMoE-1B-7B (Q4_K_M) | 8GB | Yes (naive) | *pending* | *pending* | PASS (token-for-token) |
| 4 | OLMoE-1B-7B (Q4_K_M) | 8GB | Yes (LRU cache + Strategy B) | ~10.7 | 1,181,916 kB (anon, 1 GB cache) | PASS (token-for-token) |
| 8 | ~27B MoE (Q4_K_M) | 8GB | Yes (full pipeline) | *pending* | *pending* | *pending* |

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
│   └── logs/                            # Baseline benchmark logs (gitignored)
├── scripts/                         # Build, run, and validation scripts
│   ├── build_phase3.sh
│   ├── build_phase4.sh
│   ├── run_phase3.sh
│   ├── compare_output.py                # Output validator
│   ├── inspect_gguf.py                  # GGUF file inspector
│   ├── phase2_manual_expert_read.py     # Manual expert offset verifier
│   └── validate_phase3.py               # Phase 3 output validator
├── src/cxx/                         # C++ streaming code (Phase 3+)
│   ├── phase3_stream.cpp
│   ├── phase4_stream.cpp
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