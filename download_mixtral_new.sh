#!/bin/bash
# Script de download do Mixtral-8x7B-Instruct-v0.1 (Q4_K_M) re-convertido
# GGUF novo (formato fused, compatível com llama.cpp atual)
# Repo: matteocavestri/Mixtral-8x7B-Instruct-v0.1-Q4_K_M-GGUF
# Usa a biblioteca Python huggingface_hub (huggingface-cli está deprecado)

set -e

MODEL_DIR="./models"
MODEL_FILE="mixtral-8x7b-instruct-v0.1-q4_k_m.gguf"
MODEL_REPO="matteocavestri/Mixtral-8x7B-Instruct-v0.1-Q4_K_M-GGUF"

echo "=========================================="
echo "Experimento MoE - Script de Download"
echo "=========================================="
echo ""
echo "Baixando: ${MODEL_FILE}"
echo "De: ${MODEL_REPO}"
echo "Para: ${MODEL_DIR}"
echo "Obs: ~28,4 GB"
echo ""

# Garante que o diretório de modelos exista
mkdir -p "${MODEL_DIR}"

# Verifica se Python e huggingface_hub estão disponíveis
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Erro: biblioteca huggingface_hub não instalada."
    echo "Instale com: pip install huggingface-hub"
    exit 1
fi

# Baixa o modelo usando Python/huggingface_hub
echo "Iniciando download..."
python << EOF
from huggingface_hub import hf_hub_download
import os

model_repo = "${MODEL_REPO}"
model_file = "${MODEL_FILE}"
local_dir = "${MODEL_DIR}"

try:
    downloaded_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Baixado para: {downloaded_path}")
except Exception as e:
    print(f"Erro ao baixar modelo: {e}")
    exit(1)
