#!/usr/bin/env python3
"""
Inspecionar o layout interno do GGUF do Mixtral-8x7B (re-convertido).
Detecta se os experts estão fused (tensor único com dim expert) ou separados.
"""

import os
from gguf import GGUFReader

# Resolve o caminho em relação à localização do script (funciona de qualquer diretório)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "mixtral-8x7b-instruct-v0.1-q4_k_m.gguf")

print(f"Modelo: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERRO: arquivo não encontrado. Confira com: ls -lh {os.path.join(SCRIPT_DIR, '..', 'models')}")
    raise SystemExit(1)

reader = GGUFReader(MODEL_PATH)

print("=== Architecture metadata ===")
for key in ["general.architecture", "llama.block_count", "llama.expert_count",
            "llama.expert_used_count", "llama.feed_forward_length",
            "llama.embedding_length", "general.file_type"]:
    field = reader.fields.get(key)
    val = field.contents() if field else None
    print(f"{key:35s} = {val}")

print()
print("=== Layer 0: tensores que contêm 'ffn' (detecta o layout) ===")
layer0_ffn = [t for t in reader.tensors if t.name.startswith("blk.0.ffn")]
for t in layer0_ffn:
    print(f"  {t.name:55s} shape={list(t.shape)} n_bytes={t.n_bytes}")

print()
print("=== Layout detection ===")
fused = [t for t in reader.tensors if t.name == "blk.0.ffn_gate_exps.weight"]
per_expert = [t for t in reader.tensors if t.name.startswith("blk.0.ffn_gate_exps.")]
if fused:
    print("LAYOUT: FUSED (tensor único com dim expert) -> offset = data_offset + K * bytes_per_expert")
    t = fused[0]
    print(f"  shape={t.shape} n_bytes={t.n_bytes} data_offset={t.data_offset}")
elif per_expert:
    print("LAYOUT: SEPARADO POR EXPERT (tensores blk.0.ffn_gate_exps.{i}.weight)")
    print("  -> arquivo ANTIGO, incompatível com llama.cpp atual (issue #10244)")
    for t in per_expert[:8]:
        print(f"  {t.name:50s} shape={list(t.shape)} n_bytes={t.n_bytes}")
else:
    print("Nenhum tensor ffn encontrado na layer 0 — checar nomes reais abaixo.")

print()
print("=== Todos os tensores com 'exps' no nome (amostra) ===")
exp_tensors = [t for t in reader.tensors if "exps" in t.name]
print(f"Total: {len(exp_tensors)}")
for t in exp_tensors[:12]:
    print(f"  {t.name:55s} shape={list(t.shape)} n_bytes={t.n_bytes}")

print()
print(f"Total de tensores: {len(reader.tensors)}")