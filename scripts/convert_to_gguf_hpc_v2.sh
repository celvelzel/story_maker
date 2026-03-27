#!/bin/bash
# GGUF Model Converter - For GPU HPC Servers (Pip Version)
# No cmake or build required - uses pip

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR_1="/puhome/25116696g/NLP/merged_model_qwen_2.5_3B"
MODEL_DIR_2="/puhome/25116696g/NLP/merged_model_Qwen3-4B-Instruct-2507"
OUTPUT_DIR="/puhome/25116696g/NLP/qwen-gguf"

# =============================================================================

echo "========================================"
echo "  Qwen to GGUF Converter (Pip版本)"
echo "========================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Install llama.cpp from pip
echo "[1/4] Installing llama.cpp from pip..."
pip install -q llama.cpp

# Find installation path
LLAMA_DIR=$(python -c "import llama_cpp; print(llama_cpp.__file__.replace('/__init__.py',''))")
echo "llama.cpp installed at: ${LLAMA_DIR}"

# Install dependencies
echo "[2/4] Installing dependencies..."
pip install -q numpy torch

# =============================================================================
# Convert Model 1: qwen_2.5_3B
# =============================================================================
echo "[3/4] Converting Model 1: qwen_2.5_3B"

python "${LLAMA_DIR}/convert.py" \
    "${MODEL_DIR_1}" \
    --outtype f16 \
    --outfile "${OUTPUT_DIR}/qwen2.5-3b-f16.gguf"

echo "(Quantizing to Q4...)"
python "${LLAMA_DIR}/quantize.py" \
    "${OUTPUT_DIR}/qwen2.5-3b-f16.gguf" \
    "${OUTPUT_DIR}/qwen2.5-3b-q4_0.gguf" \
    q4_0

# =============================================================================
# Convert Model 2: Qwen3-4B-Instruct-2507
# =============================================================================
echo "[4/4] Converting Model 2: Qwen3-4B-Instruct-2507"

python "${LLAMA_DIR}/convert.py" \
    "${MODEL_DIR_2}" \
    --outtype f16 \
    --outfile "${OUTPUT_DIR}/qwen3-4b-f16.gguf"

echo "(Quantizing to Q4...)"
python "${LLAMA_DIR}/quantize.py" \
    "${OUTPUT_DIR}/qwen3-4b-f16.gguf" \
    "${OUTPUT_DIR}/qwen3-4b-q4_0.gguf" \
    q4_0

# =============================================================================
echo ""
echo "========================================"
echo "  Conversion Complete!"
echo "========================================"
echo "Outputs:"
echo "  1. ${OUTPUT_DIR}/qwen2.5-3b-q4_0.gguf"
echo "  2. ${OUTPUT_DIR}/qwen3-4b-q4_0.gguf"
echo ""