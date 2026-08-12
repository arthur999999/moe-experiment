#!/bin/bash
# Download script for OLMoE-1B-7B-0924-Instruct model
# This is the baseline model for Phase 1 of the MoE Experiment

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

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed."
    echo "Please install it with: pip install huggingface-hub"
    exit 1
fi

# Download the model using huggingface-cli
huggingface-cli download "${MODEL_REPO}" \
    --include "${MODEL_FILE}" \
    --local-dir "${MODEL_DIR}"

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
