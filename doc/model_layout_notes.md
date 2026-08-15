# Model Layout Notes — OLMoE-1B-7B-0924-Instruct (Q4_K_M)

---

## Table of Contents

1. [Architecture Metadata](#architecture-metadata)
2. [Relevant Tensors per Layer](#relevant-tensors-per-layer)
3. [CPU_REPACK Memory Duplication](#cpu_repack-memory-duplication)
4. [Scale Tensors Investigation](#scale-tensors-investigation)
5. [Phase 2 Results](#phase-2-results)
6. [Performance Baseline](#performance-baseline)
7. [Peak RSS Measurements](#peak-rss-measurements)
8. [Phase 8 Inspection Results](#phase-8-inspection-results)
9. [Open Items](#open-items)
10. [Baseline Configuration Decision](#baseline-configuration-decision)
11. [Reference Output Extraction Pipeline](#reference-output-extraction-pipeline)
12. [Reference Output Portability Note](#reference-output-portability-note)

---

## Architecture Metadata

From llama.cpp load log:

- `general.architecture` = olmoe
- `olmoe.block_count` = 16          (16 layers — much smaller than larger models, good for prototyping)
- `olmoe.context_length` = 4096
- `olmoe.embedding_length` = 2048
- `olmoe.feed_forward_length` = 1024   (intermediate FFN size per expert)
- `olmoe.attention.head_count` = 16
- `olmoe.attention.head_count_kv` = 16
- `olmoe.expert_count` = 64          (total experts per layer)
- `olmoe.expert_used_count` = 8      (experts activated per token — top-8 of 64)
- `general.file_type` = 15 (Q4_K - Medium)
- model params = 6.92B total, labeled "A1.7B" (Active 1.7B)

---

## Relevant Tensors per Layer

Names confirmed in the load log:

Per layer (blk.N.*):
- `ffn_gate_inp.weight`   → router/gate (decides which experts to use)
- `ffn_gate_exps.weight`  → fused tensor, all 64 experts, gate projection
- `ffn_up_exps.weight`    → fused tensor, all 64 experts, up projection
- `ffn_down_exps.weight`  → fused tensor, all 64 experts, down projection

Tensor types in the file: 81x f32, 97x q4_K, 17x q6_K (195 tensors total)

---

## CPU_REPACK Memory Duplication

**Important finding:** llama.cpp does NOT use plain mmap only. It performs a "repack" step to optimize the memory layout for CPU matmul (q4_K_8x8), which means the weights end up duplicated in RAM:

```
  CPU_Mapped model buffer size =  4016 MiB   (mmap of the original file)
  CPU_REPACK model buffer size =  3006 MiB   (repacked copy)
```

**Result:** Total weights in RAM ≈ 7GB, even though the file itself is only ~4.2GB.

### Implication for Streaming (Phase 2+)

When comparing manual reads (offset computed from the GGUF file) against what's resident in RAM, we need to compare against the CPU_Mapped buffer (original/raw mmap), NOT the CPU_REPACK buffer (already transformed, different layout).

We also need to decide whether to disable repacking when implementing streaming — it exists purely to speed up matmul, but directly conflicts with the goal of keeping RAM usage minimal. Investigate build/runtime flag to disable it (candidate: an environment variable or flag related to GGML_CPU_REPACK — CONFIRM).

---

## Scale Tensors Investigation

### Initial Finding

The load log showed extra tensors per layer, e.g.:
- `blk.N.ffn_gate_exps.scale`
- `blk.N.ffn_gate_exps.input_scale`
- (same pattern for attn_q, attn_k, attn_v, attn_output, ffn_down_exps, ffn_up_exps)

### Resolution (via Phase 8)

**RESOLVED:** `.scale` / `.input_scale` tensors do NOT exist in the file.

Searched all 195 tensors for "scale" in the name — found ZERO matches. The tensors seen in llama.cpp's load log are runtime-allocated buffers (likely part of a generic dynamic quantization code path that exists regardless of architecture, left unused/empty for pre-quantized GGUF).

**Implication:** These do NOT need to be accounted for when calculating per-expert byte offsets — they don't come from the file.

---

## Performance Baseline

> **Superseded**: these numbers used the default (repack-enabled)
> config, before the `--no-repack` decision was made. See
> "Baseline Configuration Decision" below — `--no-repack` is the
> official baseline going forward. Kept here for historical record only.

Phase 1 results with fixed config: `-n 128 -t 4 --temp 0 --top-k 1 --seed 42 -no-cnv -st`

| Run    | Prompt t/s | Generation t/s |
|--------|-----------|-----------------|
| 1      | 70.0      | 18.8            |
| 2      | 77.7      | 19.0            |
| 3      | 72.4      | 17.7            |
| 4      | 77.5      | 18.3            |
| 5 (with time -v) | 63.2 | 18.3       |

**Averages:**
- Generation: ~18.4 t/s (variance ~7%)
- Prompt: ~72.2 t/s

**Generated output:** identical across all runs (determinism confirmed)

---

## Peak RSS Measurements

> **Superseded**: measured with the default (repack-enabled) config.
> See "Baseline Configuration Decision" below for the current official
> number (4.68 GB peak RSS with `--no-repack`).

Measured via `/usr/bin/time -v`:

**Maximum resident set size:** 7,762,304 KB ≈ 7.4 GB

### Expected Breakdown (from the load log):

| Component        | Size     |
|------------------|----------|
| mmap (CPU_Mapped)     | ≈ 4016 MiB |
| repack (CPU_REPACK)   | ≈ 3006 MiB |
| KV cache              | 512 MiB  |
| compute buffer        | 68 MiB   |
| **Total**             | **≈ 7602 MiB** |

(Matches the measured ~7.4GB)

### Conclusion

Even a "small" model (6.9B total / 1.7B active) already uses nearly the entire 8GB budget WITHOUT streaming, mostly due to the repack duplication. This reinforces why streaming (and possibly disabling repack) is necessary even for models this size, not just for the larger ones in the roadmap.

---

## Phase 8 Inspection Results

### Tensor Count Sanity Check

Total tensors read via gguf-py: **195** — matches llama.cpp's load log ("loaded meta data with 43 key-value pairs and 195 tensors"). Confirms gguf-py is reading the same file correctly.

### Shape Order Convention

**Confirmed:** shape order is ggml convention (ne[0] fastest-varying). gguf-py reports tensor.shape in ggml's [ne0, ne1, ne2, ...] order — NOT numpy/C convention. The expert axis is NOT always shape[0]; its position must be located by matching against the known expert_count (64), not assumed.

### Per-Layer Expert Tensors (Layer 0 Reference)

#### blk.0.ffn_gate_exps.weight

| Attribute | Value |
|-----------|-------|
| Shape (ggml order) | [2048, 1024, 64] → [n_embd, n_ff, n_expert] |
| Expert axis index | 2 |
| Tensor type | 12 (Q4_K) |
| Total bytes | 75,497,472 |
| Bytes per expert | 1,179,648 (n_bytes / 64) |
| Data offset (file) | 169,841,472 |
| **Offset formula** | `offset(expert K) = 169,841,472 + K * 1,179,648` |

#### blk.0.ffn_up_exps.weight

| Attribute | Value |
|-----------|-------|
| Shape (ggml order) | [2048, 1024, 64] → same layout as gate |
| Tensor type | 12 (Q4_K) |
| Total bytes | 75,497,472 |
| Bytes per expert | 1,179,648 |
| Data offset (file) | 245,338,944 |
| **Offset formula** | `offset(expert K) = 245,338,944 + K * 1,179,648` |

#### blk.0.ffn_down_exps.weight

| Attribute | Value |
|-----------|-------|
| Shape (ggml order) | [1024, 2048, 64] → [n_ff, n_embd, n_expert] |
| Expert axis index | 2 |
| Notes | Axis order REVERSED vs gate/up (down projects ff→embd, gate/up project embd→ff). Same n_elements by coincidence, but do NOT assume down always matches gate/up for other models. |
| Tensor type | 14 (Q6_K) |
| Total bytes | 110,100,480 |
| Bytes per expert | 1,720,320 (n_bytes / 64) |
| Data offset (file) | 59,740,992 |
| **Offset formula** | `offset(expert K) = 59,740,992 + K * 1,720,320` |

### Quantization Types Confirmed

| Type Code | Quantization | Used By |
|-----------|-------------|---------|
| 12 | Q4_K | ffn_gate_exps, ffn_up_exps |
| 14 | Q6_K | ffn_down_exps |

This matches the file-level breakdown (97 tensors q4_K, 17 tensors q6_K, 81 tensors f32) and is a common mixed-quantization choice — the down-projection is often kept at higher precision (Q6_K) since it tends to be more sensitive to quantization error.

### Implication for Phase 2

When reading raw expert slices from disk:
- Gate/up experts must be interpreted as **Q4_K** blocks
- Down experts must be interpreted as **Q6_K** blocks

These have different block sizes/layouts internally, already reflected in the differing bytes-per-expert (1,179,648 for Q4_K vs 1,720,320 for Q6_K). The actual in-block byte structure must be read from ggml's quantization format definitions before writing a decoder.

---

## Phase 2 Results

### Correctness Gate: PASSED (36/36)

Tested 3 layers (0, 8, 15) × 3 tensor kinds (gate/up/down) × 4 expert indices (0, 1, 31, 63) = 36 combinations. For each:
1. Computed the expert's file offset independently from GGUF metadata
2. Verified offset/size land on exact quantization block boundaries
3. Read exact byte range directly from raw file via `open()/seek()/read()`
4. Compared against equivalent slice from gguf-py's tensor data
5. All 36 matched byte-for-byte

### Key Correction: Quantization Types Vary Per-Layer

Earlier assumption that ffn_down_exps is always Q6_K and ffn_gate/up_exps are always Q4_K is **WRONG**. Phase 2 testing revealed:

| Layer | Tensor | Quantization | Bytes/Expert |
|-------|--------|-------------|--------------|
| 0 | ffn_down_exps | Q6_K | 1,720,320 |
| 8 | ffn_down_exps | Q4_K | 1,179,648 |
| 15 | ffn_down_exps | Q6_K | 1,720,320 |

Gate/up_exps were Q4_K in all tested layers, but should not be assumed to hold for every layer. This is likely due to per-tensor importance-matrix-guided quantization.

**Implication for Phase 3:** Any code reading expert tensors MUST look up the quantization type per (layer, tensor_kind) individually via GGUF metadata — never hardcode an assumed quant type per tensor kind.

### gguf-py tensor.data Structure Note

Discovered and documented during Phase 2: `tensor.data` for quantized types is NOT a flat byte buffer. It's reshaped to reversed-ggml-axis order with the last axis converted from elements to bytes. Since the expert axis is the outermost ggml axis, after reversal it becomes axis 0 of the numpy array — so per-expert data is obtained via `tensor.data[expert_idx]`, NOT by computing a flat byte-range slice.

Using flat byte slicing (`tensor.data[start:end]`) silently produces wrong results (returns whole array for expert 0 due to Python's slice clamping, empty arrays for others) without raising errors — a dangerous silent-failure mode to watch for.

### Process Note: Verification Must Be Repeated Per Model

Everything discovered in Phase 2 (offset formula behavior, block alignment, and especially the per-layer quantization variability) is **specific to THIS file**. Different architectures, different quantization schemes, or even a different quantization of the same model may:

- Use a different number of tensors per expert
- Put the expert axis in a different shape position
- Use different (or more varied) quantization types per tensor
- Not have a clean divisor between n_bytes and n_expert at all

`scripts/phase2_manual_expert_read.py` was written to discover these properties from metadata rather than assume them, and includes explicit guards (e.g. raises if the expert axis isn't where expected).

**Mandatory Onboarding Step:** Going forward, running this verification script against a NEW model — and getting a clean PASS on all sampled combinations — is a **mandatory** step before trusting any offset math for that model, not an optional sanity check.

**Applies to:** Every model introduced in Phase 8:
- Qwen1.5-MoE-A2.7B
- Qwen3-30B-A3B
- Final ~27B target

---

## Open Items

### General

- [x] Confirm byte offset of each tensor within the file (DONE)
- [x] Calculate bytes per individual expert (stride) within fused tensors (DONE)
- [x] Understand the purpose of .scale / .input_scale tensors (RESOLVED: runtime buffers, not in file)
- [x] Confirm whether a flag exists to disable CPU_REPACK (DONE: `--no-repack`)

---

## Baseline Configuration Decision

**Decision:** Use `--no-repack` as the standard baseline config.

### Comparison

Measured on this hardware (x86_64, Manjaro):

| Config       | Peak RSS | Generation t/s (avg) |
|--------------|----------|------------------------|
| repack (default) | 7.4 GB   | ~18.4 t/s        |
| --no-repack  | 4.68 GB  | ~20.5 t/s              |

**Repack runs** (n=5): 18.8, 19.0, 17.7, 18.3, 18.3 t/s
**No-repack runs** (n=3): 21.3, 20.5, 19.7 t/s

### Determinism Notes

Both configs are internally deterministic (identical output across repeated runs), but generate DIFFERENT text from each other — repack changes the compute kernel path, which shifts floating-point reduction order enough to change greedy-decoded output token by token.

**Implication:** repack on/off must be treated as part of the FIXED baseline config, same as thread count and seed. Changing it mid-project would silently break the correctness gate.

### Rationale

**Adopt --no-repack as the standard flag for all Phase 1+ runs.**

Lower RAM usage aligns directly with the project's goal (fit large models in a small RAM budget), and it also happened to be faster on this specific hardware — not a given, possibly x86-specific or specific to this model's small ffn size, but empirically true here.

### Open Question

Why no-repack is faster here is not fully understood. Candidate explanations:
- Repack may be tuned more for ARM (AARCH64_REPACK online repacking)
- The repack overhead may not pay off on a model this small (n_ff=1024, 16 layers)
- The larger resident memory footprint with repack may hurt cache locality more than the repacked kernel helps

Not investigated further — out of scope for now.

---

## Reference Output Extraction Pipeline

The `reference_output.txt` file is the **correctness gate** for every future phase (streaming implementations must match this byte-for-byte, or in practice text-for-text given ANSI/whitespace stripping).

### Fixed Configuration

```bash
-p "Explain how mixture of experts routing works." -n 128 -t 4 \
  --temp 0 --top-k 1 --seed 42 -no-cnv -st --no-repack
```

### Extraction Pipeline

**Problem:** llama-cli's stdout includes an ASCII banner, ANSI color codes, and prompt echo even with `--simple-io --log-disable`, so raw stdout is not usable directly as a reference.

**Solution:**

**Step 1:** Generate raw output
```bash
./llama.cpp/build/bin/llama-cli \
  -m ./models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf \
  -p "Explain how mixture of experts routing works." \
  -n 128 -t 4 --temp 0 --top-k 1 --seed 42 -no-cnv -st --no-repack \
  --simple-io --log-disable \
  > doc/reference_output_raw.txt
```

**Step 2:** Clean and extract generated text
```bash
sed 's/\x1b\[[0-9;]*m//g' doc/reference_output_raw.txt \
  | awk '/^> Explain/{flag=1; next} /^\[ Prompt:/{flag=0} flag' \
  | sed '/^$/N;/^\n$/D' \
  > doc/reference_output.txt
```

**Step 3:** Normalize trailing whitespace to match `scripts/compare_output.py`
```bash
python3 -c "
content = open('doc/reference_output.txt').read()
content = content.strip() + '\n'
open('doc/reference_output.txt', 'w').write(content)
"
```

Without this step, `scripts/compare_output.py` reports a false MISMATCH due to
a trailing blank line left by the sed/awk pipeline above, even though
the generated text content is identical. This bit us during Phase 1
validation — see the script's `clean_output()` for the matching
normalization on the "current" side.

### Important Note

The awk filter matches on `^> Explain` specifically — if the prompt text changes in a future run, this filter needs updating to match the new prompt's echo line.

---

## Reference Output Portability Note

`reference_output.txt` IS committed to git — it's the correctness gate used by `scripts/compare_output.py`. However, greedy-decoded output is only guaranteed reproducible on the exact hardware + llama.cpp build that generated it.

**Proof:** `--repack` vs `--no-repack` alone changes output on the SAME machine, due to floating-point reduction order differences.

**Implication:** If this project moves to different hardware or a newer llama.cpp build, `reference_output.txt` likely needs to be regenerated before `scripts/compare_output.py` can be trusted again.

---
