# Phase 3 -- Naive Synchronous Expert Streaming (RESULTS)

**Status:** COMPLETE -- correctness gate PASSED

## Summary

Phase 3 implements naive synchronous expert streaming for OLMoE-1B-7B (Q4_K_M) on llama.cpp commit `0b1bad1`. Routed experts are read from disk on demand (via `pread()`) instead of relying on mmap, and the generated output is **token-for-token identical** to the resident baseline.

The streaming implementation hooks llama.cpp's public `cb_eval` (a `ggml_backend_sched_eval_callback`) and, on each router node, reads exactly the top-8 experts that token selected in that layer, then rebinds the expert tensors' `->data` to buffers filled from disk.

## Exit Criterion

> Streamed inference on OLMoE-1B-7B produces output identical to the resident baseline.

**Result: PASSED.** Streaming ON vs Streaming OFF (plain mmap) produce identical tokens. The streamer also matches the freshly regenerated reference output token-for-token.

## Files

- `src/cxx/phase3_stream.cpp` -- the streamer (main + callback + pread + rebind)
- `src/cxx/debug_graph.cpp` -- graph debug / discovery helper
- `scripts/build_phase3.sh` -- build against the already-compiled llama.cpp
- `scripts/run_phase3.sh` -- runs with fixed config, separates stdout (text) / stderr (logs)
- `scripts/validate_phase3.py` -- compares streamer output vs reference (whitespace-normalized)

## Configuration

Fixed (same spirit as Phase 1 baseline):

```
model : OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
prompt : "Explain how mixture of experts routing works."
n_predict: 128
threads : 4
seed : 42
greedy : temp 0
repack : OFF (use_extra_bufts = false)
```

## Telemetry (validation run)

```
16 layers, 64 experts, top-8
[DIAG] layer0.gate data_offset=169841472 (expected 169841472)
warm-up: 32/32 experts discovered
Phase 3: 48768 reads, 128 tokens
```

Reads check: `(128 - 1) decodes x 16 layers x 8 experts x 3 tensors = 48768`

## Key Findings (important for later phases)

### 1. GGUF offsets in C -- the classic gotcha

`gguf_get_tensor_offset()` returns the tensor offset **relative to the tensor data section**, NOT the absolute file offset. The absolute offset requires adding the base:

```c
offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tensor_id);
```

Verified: `blk.0.ffn_gate_exps.weight` -> `169841472` (matches the Phase 2 Python result). Without the base, every `pread()` reads from a shifted position and **every byte diverges** (diagnosed as the everything-token-0 bug).

### 2. Only observe the router node (efficient)

`ggml_backend_sched_eval_callback(t, ask, ud)`:
- When `ask == true`, the scheduler asks if you want to observe a node. Returning `false` lets it batch the node together with others (you never see it).
- After warm-up, we return `true` **only** for router nodes (`ffn_moe_topk`); everything else is batched. This is the efficiency win shared with BigMoeOnEdge.

### 3. Router node names carry the layer

`ffn_moe_topk-%d` -- the layer index is parsed directly from the node name via `sscanf`.

### 4. Expert weights appear as `src`, not graph nodes

During warm-up we discover expert tensors by scanning each node's `src[i]` for names containing `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` -- they are sources of the `mul_mat_id` (MoE) ops, never visited as standalone nodes.

### 5. The router buffer layout

The router node has shape `[8, 1, 1, 1]` (single-token decode), holding the top-8 expert ids as `int32_t`. Strategy A (full-size scratch buffer per tensor kind, experts scattered to their correct absolute offsets) is used -- correct and simple for Phase 3. Compact id-remap (Strategy B) is deferred to Phase 4+ with the cache.

### 6. Chat template must match the reference

The prompt must be formatted with the model's chat template (via `llama_model_meta_val_str("tokenizer.chat_template")` + `llama_chat_apply_template`) to match how reference outputs are generated. Without it, the very first token differs.

### 7. Streaming is not the cause of reference mismatch

Controlled experiment: Streaming ON and OFF produce identical output. Any divergence vs an externally generated reference is a config/prompt difference (chat template, context, batching), not a streaming bug. The reference output must be regenerated on the current build/config to be a valid gate.

## Correctness Evidence

| Check | Result |
|-------|--------|
| Expert offset (layer 0 gate) | 169841472 (PASS) |
| Warm-up discovery (gate/down per layer) | 32/32 (PASS) |
| pread bytes == mmap bytes | PASS (OK: expert == mmap) |
| Streaming ON == Streaming OFF | PASS (identical tokens) |
| Streamer output == reference | PASS (token-by-token) |