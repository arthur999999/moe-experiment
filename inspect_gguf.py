#!/usr/bin/env python3
"""
Phase 1, Step 8: inspect the internal layout of the OLMoE GGUF file.
"""

from gguf import GGUFReader

MODEL_PATH = "./models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"

reader = GGUFReader(MODEL_PATH)

print("=== Architecture metadata ===")
for key in ["general.architecture", "olmoe.block_count", "olmoe.expert_count",
            "olmoe.expert_used_count", "olmoe.feed_forward_length",
            "olmoe.embedding_length"]:
    field = reader.fields.get(key)
    val = field.contents() if field else None
    print(f"{key:35s} = {val}")

print()
print("=== Layer 0 expert tensors (fused, all experts) ===")
target_names = ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"]

for tensor in reader.tensors:
    if tensor.name in target_names:
        print(f"\nname:        {tensor.name}")
        print(f"shape:       {tensor.shape}   (ggml order: ne[0], ne[1], ne[2]...)")
        print(f"tensor_type: {tensor.tensor_type}")
        print(f"n_elements:  {tensor.n_elements}")
        print(f"n_bytes:     {tensor.n_bytes}")
        print(f"data_offset: {tensor.data_offset}")

print()
print("=== Per-expert size calculation (layer 0, ffn_gate_exps) ===")
N_EXPERT = 64  # confirmed from metadata above
for tensor in reader.tensors:
    if tensor.name == "blk.0.ffn_gate_exps.weight":
        expert_axis_index = list(tensor.shape).index(N_EXPERT)
        bytes_per_expert = tensor.n_bytes / N_EXPERT
        print(f"Full tensor shape (ggml order): {tensor.shape}")
        print(f"Expert axis found at index: {expert_axis_index} (value={N_EXPERT})")
        print(f"Total tensor bytes: {tensor.n_bytes}")
        print(f"Bytes per expert: {bytes_per_expert}")
        print(f"Base file offset of this tensor: {tensor.data_offset}")
        print(f"  -> offset of expert K = data_offset + K * {bytes_per_expert}")

print()
print("=== Searching ALL tensor names containing 'scale' (any block) ===")
scale_tensors = [t for t in reader.tensors if "scale" in t.name]
print(f"Found {len(scale_tensors)} tensors with 'scale' in the name")
for t in scale_tensors[:10]:
    print(f"  {t.name:45s} shape={t.shape} n_bytes={t.n_bytes}")

print()
print("=== Total tensor count and first/last 5 names (sanity check) ===")
all_names = [t.name for t in reader.tensors]
print(f"Total tensors: {len(all_names)}")
print("First 5:", all_names[:5])
print("Last 5:", all_names[-5:])
