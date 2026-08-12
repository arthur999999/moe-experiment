# Model Layout Notes — OLMoE-1B-7B-0924-Instruct (Q4_K_M)

---

## Table of Contents

1. [Architecture Metadata](#architecture-metadata)
2. [Relevant Tensors per Layer](#relevant-tensors-per-layer)
3. [CPU_REPACK Memory Duplication](#cpu_repack-memory-duplication)
4. [Scale Tensors (Pending Investigation)](#scale-tensors-pending-investigation)
5. [Performance Baseline](#performance-baseline)
6. [Peak RSS Measurements](#peak-rss-measurements)
7. [Open Items for Phase 8](#open-items-for-phase-8)
8. [Baseline Configuration Decision](#baseline-configuration-decision)
9. [Reference Output Extraction Pipeline](#reference-output-extraction-pipeline)

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

## Scale Tensors (Pending Investigation)

**Not fully understood yet.**

The load log shows extra tensors per layer, e.g.:
- `blk.N.ffn_gate_exps.scale`
- `blk.N.ffn_gate_exps.input_scale`
- (same pattern for attn_q, attn_k, attn_v, attn_output, ffn_down_exps, ffn_up_exps)

We haven't confirmed the exact purpose of these extra tensors in this specific file yet (possibly related to imatrix-based quantization, which the metadata confirms is present: quantize.imatrix.file, quantize.imatrix.dataset, quantize.imatrix.entries_count = 128).

**Action:** INVESTIGATE in Phase 8 (inspection via gguf-py) before calculating per-expert offsets — these scale tensors also need to be accounted for if they affect per-expert size.

---

## Performance Baseline

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

## Open Items for Phase 8

Investigation tasks via gguf-py:

- [ ] Confirm the exact shape of the fused tensor `ffn_gate_exps.weight`
      (expected: something like [n_expert=64, n_ff=1024, n_embd=2048], but
      CONFIRM the exact axis order by reading it via gguf-py)
- [ ] Confirm the byte offset of each tensor within the file
- [ ] Calculate bytes per individual expert (stride) within the fused tensor
- [ ] Understand the purpose of the .scale / .input_scale tensors and
      whether they affect the per-expert offset calculation
- [ ] Confirm whether a flag exists to disable CPU_REPACK at build/runtime

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
  > reference_output_raw.txt
```

**Step 2:** Clean and extract generated text
```bash
sed 's/\x1b\[[0-9;]*m//g' reference_output_raw.txt \
  | awk '/^> Explain/{flag=1; next} /^\[ Prompt:/{flag=0} flag' \
  | sed '/^$/N;/^\n$/D' \
  > reference_output.txt
```

### Important Note

The awk filter matches on `^> Explain` specifically — if the prompt text changes in a future run, this filter needs updating to match the new prompt's echo line.
