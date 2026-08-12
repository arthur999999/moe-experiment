#!/bin/bash
# Download script for OLMoE-1B-7B-0924-Instruct model
# This is the baseline model for Phase 1 of the MoE Experiment
# Uses huggingface_hub Python library instead of deprecated huggingface-cli

set -e

MODEL_DIR="./models"
MODEL_FILE="OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
MODEL_REPO="bartowski/OLMoE-1B-7B-0924-Instruct-GGUF"

echo "=========================================="
echo "MoE Experiment - Model Download Script"
echo "=========================================="
echo ""
echo "Downloading: ${MODEL_FILE}"
echo "From: ${MODEL_REPO}"
echo "To: ${MODEL_DIR}"
echo ""

# Ensure models directory exists
mkdir -p "${MODEL_DIR}"

# Check if Python and huggingface_hub are available
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Error: huggingface_hub Python library is not installed."
    echo "Please install it with: pip install huggingface-hub"
    exit 1
fi

# Download the model using Python/huggingface_hub
echo "Starting download..."
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
    print(f"Downloaded to: {downloaded_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)
EOF

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Model saved to: ${MODEL_DIR}/${MODEL_FILE}"
echo ""
echo "Next steps:"
echo "1. Build llama.cpp: cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release && cmake --build llama.cpp/build --config Release -j\$(nproc)"
echo "2. Run Phase 1 baseline: ./llama.cpp/build/bin/llama-cli -m ${MODEL_DIR}/${MODEL_FILE} ..."
echo ""
