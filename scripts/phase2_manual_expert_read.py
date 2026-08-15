#!/usr/bin/env python3
"""
Phase 2: manual expert offset calculation and verification.

For a sample of (layer, tensor, expert_index) combinations, this script:
  1. Computes the expert's byte offset and size purely from GGUF metadata
     (no reliance on gguf-py's own tensor-slicing logic).
  2. Confirms that offset/size land on exact quantization block boundaries
     (Q4_K: 256 elements / 144 bytes per block; Q6_K: 256 elements / 210
     bytes per block) — a cut that doesn't land on a block boundary would
     produce undecodable data.
  3. Reads that exact byte range directly from the raw .gguf file via
     open()/seek()/read() — completely independent of gguf-py's tensor
     loading path.
  4. Reads the same logical slice via gguf-py's own tensor.data array
     (already mmap-loaded).
  5. Compares the two byte-for-byte.

If all comparisons match, our offset math is proven correct without
needing llama.cpp or any inference to be involved.
"""

import sys
from gguf import GGUFReader, GGML_QUANT_SIZES, GGMLQuantizationType

MODEL_PATH = "../models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
N_EXPERT = 64

TENSOR_KINDS = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
LAYERS_TO_TEST = [0, 8, 15]
EXPERTS_TO_TEST = [0, 1, 31, 63]


def get_expert_slice_info(reader, layer, tensor_kind, expert_idx):
    """Compute offset (absolute, in-file) and byte size for one expert's
    slice within a fused expert tensor. Returns a dict with everything
    needed to both read the file directly and to slice tensor.data.
    """
    tensor_name = f"blk.{layer}.{tensor_kind}.weight"
    tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        raise ValueError(f"Tensor not found: {tensor_name}")

    shape = list(tensor.shape)
    if N_EXPERT not in shape:
        raise ValueError(f"{tensor_name}: expert count {N_EXPERT} not found in shape {shape}")
    expert_axis = shape.index(N_EXPERT)
    if expert_axis != len(shape) - 1:
        raise ValueError(
            f"{tensor_name}: expected expert axis to be the last ggml axis, "
            f"found it at index {expert_axis} of shape {shape}. "
            f"Offset math below assumes experts are the outermost (slowest-varying) "
            f"axis — verify before proceeding."
        )

    bytes_per_expert = tensor.n_bytes // N_EXPERT
    if tensor.n_bytes % N_EXPERT != 0:
        raise ValueError(f"{tensor_name}: n_bytes {tensor.n_bytes} not evenly divisible by {N_EXPERT} experts")

    quant_type = GGMLQuantizationType(tensor.tensor_type)
    block_size_elems, block_size_bytes = GGML_QUANT_SIZES[quant_type]

    elements_per_expert = tensor.n_elements // N_EXPERT
    if elements_per_expert % block_size_elems != 0:
        raise ValueError(
            f"{tensor_name}: {elements_per_expert} elements/expert not a multiple of "
            f"{quant_type.name} block size ({block_size_elems} elements) — "
            f"a per-expert cut would land mid-block."
        )
    blocks_per_expert = elements_per_expert // block_size_elems
    expected_bytes_per_expert = blocks_per_expert * block_size_bytes
    if expected_bytes_per_expert != bytes_per_expert:
        raise ValueError(
            f"{tensor_name}: block-based byte calc ({expected_bytes_per_expert}) "
            f"disagrees with n_bytes/N_EXPERT ({bytes_per_expert})"
        )

    file_offset = tensor.data_offset + expert_idx * bytes_per_expert

    return {
        "tensor_name": tensor_name,
        "quant_type": quant_type.name,
        "shape": shape,
        "expert_axis": expert_axis,
        "bytes_per_expert": bytes_per_expert,
        "blocks_per_expert": blocks_per_expert,
        "block_size_bytes": block_size_bytes,
        "file_offset": file_offset,
        "expert_idx": expert_idx,
        "tensor": tensor,
    }


def read_from_raw_file(model_path, file_offset, n_bytes):
    with open(model_path, "rb") as f:
        f.seek(file_offset)
        return f.read(n_bytes)


def read_from_gguf_tensor_data(info):
    """IMPORTANT: gguf-py's tensor.data is NOT a flat byte buffer. For
    quantized types it's reshaped to reversed-ggml-order dims with the
    last axis converted to bytes (see GGUFReader._build_tensors). Since
    we've already verified the expert axis is the outermost ggml axis,
    it ends up as axis 0 of this numpy array after the reversal — so we
    index it directly rather than computing any flat byte offset.
    """
    tensor = info["tensor"]
    expert_slice = tensor.data[info["expert_idx"]]
    return expert_slice.tobytes()


def main():
    print(f"Loading {MODEL_PATH} ...")
    reader = GGUFReader(MODEL_PATH)
    print("Loaded.\n")

    total_checks = 0
    total_pass = 0
    failures = []

    for layer in LAYERS_TO_TEST:
        for tensor_kind in TENSOR_KINDS:
            for expert_idx in EXPERTS_TO_TEST:
                total_checks += 1
                label = f"layer={layer:2d} tensor={tensor_kind:15s} expert={expert_idx:2d}"
                try:
                    info = get_expert_slice_info(reader, layer, tensor_kind, expert_idx)
                    raw_bytes = read_from_raw_file(MODEL_PATH, info["file_offset"], info["bytes_per_expert"])
                    gguf_bytes = read_from_gguf_tensor_data(info)

                    if raw_bytes == gguf_bytes:
                        total_pass += 1
                        print(f"  PASS  {label}  ({info['quant_type']}, "
                              f"{info['bytes_per_expert']} bytes, "
                              f"{info['blocks_per_expert']} blocks)")
                    else:
                        failures.append((label, "byte mismatch"))
                        print(f"  FAIL  {label}  <-- byte mismatch "
                              f"(raw={len(raw_bytes)}B, gguf={len(gguf_bytes)}B)")
                except Exception as e:
                    failures.append((label, str(e)))
                    print(f"  ERROR {label}  <-- {e}")

    print(f"\n{total_pass}/{total_checks} checks passed.")
    if failures:
        print("\nFailures:")
        for label, reason in failures:
            print(f"  - {label}: {reason}")
        sys.exit(1)
    else:
        print("\nAll manual offset calculations match gguf-py's own tensor data.")
        print("Phase 2 correctness gate: PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()