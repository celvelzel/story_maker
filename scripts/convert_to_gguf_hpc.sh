#!/bin/bash
# GGUF Model Converter - For GPU HPC Servers
# Run this on a machine with GPU for fastest conversion

set -e

# =============================================================================
# CONFIGURATION - Adjust paths as needed
# =============================================================================

# Model directories
MODEL_DIR_1="/puhome/25116696g/NLP/merged_model_qwen_2.5_3B"
MODEL_DIR_2="/puhome/25116696g/NLP/merged_model_Qwen3-4B-Instruct-2507"

# Output directory
OUTPUT_DIR="/puhome/25116696g/NLP/qwen-gguf"

# Where to download llama.cpp
LLAMA_CPP_DIR="/puhome/25116696g/NLP/llama.cpp"

# =============================================================================

echo "========================================"
echo "  Qwen to GGUF Converter (GPU加速版)"
echo "========================================"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Download llama.cpp (using wget - no git required)
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo "[1/5] Downloading llama.cpp..."
    wget -q https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip -O llama.cpp.zip
    unzip -q llama.cpp.zip
    mv llama.cpp-master "${LLAMA_CPP_DIR}"
    rm llama.cpp.zip
    echo "Download complete!"
else
    echo "[1/5] llama.cpp already exists, skipping..."
fi

# Install dependencies
echo "[2/5] Installing Python dependencies..."
pip install -q numpy torch

# Build llama.cpp with GPU support (BLAS)
echo "[3/5] Building llama.cpp with GPU support..."
cd "${LLAMA_CPP_DIR}"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUBLAS=ON
cmake --build build --config Release -j$(nproc)

# =============================================================================
# Convert Model 1: qwen_2.5_3B (your fine-tuned model)
# =============================================================================
echo "[4/5] Converting Model 1: qwen_2.5_3B"
echo "(This takes ~2-5 minutes on GPU)"

python "${LLAMA_CPP_DIR}/convert.py" \
    "${MODEL_DIR_1}" \
    --outtype f16 \
    --outfile "${OUTPUT_DIR}/qwen2.5-3b-f16.gguf"

# Quantize to Q4
echo "Quantizing to INT4 (Q4_0)..."
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
    "${OUTPUT_DIR}/qwen2.5-3b-f16.gguf" \
    "${OUTPUT_DIR}/qwen2.5-3b-q4_0.gguf" \
    q4_0

# =============================================================================
# Convert Model 2: Qwen3-4B-Instruct-2507
# =============================================================================
echo "[5/5] Converting Model 2: Qwen3-4B-Instruct-2507"
echo "(This takes ~2-5 minutes on GPU)"

python "${LLAMA_CPP_DIR}/convert.py" \
    "${MODEL_DIR_2}" \
    --outtype f16 \
    --outfile "${OUTPUT_DIR}/qwen3-4b-f16.gguf"

# Quantize to Q4
echo "Quantizing to INT4 (Q4_0)..."
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
    "${OUTPUT_DIR}/qwen3-4b-f16.gguf" \
    "${OUTPUT_DIR}/qwen3-4b-q4_0.gguf" \
    q4_0

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "========================================"
echo "  Conversion Complete!"
echo "========================================"
echo "Outputs:"
echo "  1. ${OUTPUT_DIR}/qwen2.5-3b-q4_0.gguf"
echo "  2. ${OUTPUT_DIR}/qwen3-4b-q4_0.gguf"
echo ""
echo "Copy these files to your Windows PC and run with llama.cpp"
echo ""
echo "To run on Windows:"
echo "  llama-cli.exe -m models\\qwen-gguf\\qwen2.5-3b-q4_0.gguf -c 2048 -n 512"
echo ""