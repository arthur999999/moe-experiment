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

**Exit criteria:** reproducible baseline numbers, a saved reference output, and a documented map of the expert tensor layout.

### Phase 2 — Manual expert reads
Write a standalone reader that, given a (layer, expert index) pair, computes the correct byte offset inside the GGUF file and reads that slice directly from disk — without going through llama.cpp yet.
- Verify each manually-read expert slice matches, byte-for-byte, the same slice already loaded by mmap
- No inference happens in this phase — this is purely about proving the offset math is correct

**Exit criteria:** manual reads match mmap-loaded data exactly, across multiple layers and experts.

### Phase 3 — Naive synchronous streaming
Hook llama.cpp's evaluation callback to intercept the router's top-k expert selection before the expert matmul runs, and substitute a just-in-time disk read for each selected expert instead of relying on mmap.
- No caching, no parallelism — one token, one layer, one expert read at a time
- Compare generated output against the Phase 1 reference output — must match exactly

**Exit criteria:** streamed inference on OLMoE-1B-7B produces output identical to the resident baseline. Speed is expected to be poor at this stage — correctness comes first.

### Phase 4 — Expert cache
Add an LRU (or similar) cache of recently-used experts in RAM, sized either manually or automatically based on available memory.
- Measure cache hit rate and its effect on tokens/sec
- Re-validate against the reference output after every change

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
```

### Building llama.cpp (baseline, Phase 1)

```bash
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j$(nproc)
```

## Benchmarks

Populated as each phase completes. No numbers are published here until they've been measured on real hardware with a fixed, documented configuration (device, thread count, prompt, seed).

| Phase | Model | RAM budget | Streaming | tok/s | Peak RSS | Output verified vs. reference |
|-------|-------|-----------|-----------|-------|----------|-------------------------------|
| 1 | OLMoE-1B-7B (Q4_K_M) | 8GB | No (baseline) | *pending* | *pending* | N/A (this **is** the reference) |
| 3 | OLMoE-1B-7B (Q4_K_M) | 8GB | Yes (naive) | *pending* | *pending* | *pending* |
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

## Acknowledgments

- The [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) project, whose architecture and benchmark methodology this project's plan is directly modeled on
- Hugging Face Hub for model hosting
- The llama.cpp community for the GGUF format and inference engine
- The OLMoE, Qwen, and Gemma teams for open MoE model releases

---

**Note:** This is experimental research software under active development. Every benchmark number in this README reflects an actual measured run, or is explicitly marked `*pending*` until it is.