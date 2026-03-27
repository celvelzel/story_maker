#!/bin/bash
# GGUF Model Converter - For GPU HPC Servers
# Downloads conversion scripts directly

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR_1="/puhome/25116696g/NLP/merged_model_qwen_2.5_3B"
MODEL_DIR_2="/puhome/25116696g/NLP/merged_model_Qwen3-4B-Instruct-2507"
OUTPUT_DIR="/puhome/25116696g/NLP/qwen-gguf"
SCRIPT_DIR="/puhome/25116696g/NLP/llama-scripts"

# =============================================================================

echo "========================================"
echo "  Qwen to GGUF Converter (直接下载版)"
echo "========================================"
echo ""

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRIPT_DIR}"

# Download convert.py and quantize.py directly
if [ ! -f "${SCRIPT_DIR}/convert.py" ]; then
    echo "[1/3] Downloading conversion scripts..."
    
    # Download convert.py
    wget -q -O "${SCRIPT_DIR}/convert.py" \
        https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert.py
    
    # Download requirements
    wget -q -O "${SCRIPT_DIR}/requirements.txt" \
        https://raw.githubusercontent.com/ggerganov/llama.cpp/master/requirements.txt
    
    # Download quantize.py
    wget -q -O "${SCRIPT_DIR}/quantize.py" \
        https://raw.githubusercontent.com/ggerganov/llama.cpp/master/quantize.py
    
    echo "Download complete!"
fi

# Install dependencies
echo "[2/3] Installing Python dependencies..."
pip install -q numpy torch transformers

# Add sentencepiece if available
pip install -q sentencepiece 2>/dev/null || true

# =============================================================================
# Convert Model 1: qwen_2.5_3B
# =============================================================================
echo "[3/3] Converting Model 1: qwen_2.5_3B"

python "${SCRIPT_DIR}/convert.py" \
    "${MODEL_DIR_1}" \
    --outtype f16 \
    --outfile "${OUTPUT_DIR}/qwen2.5-3b-f16.gguf"

echo "(Quantizing to Q4...)"
python "${SCRIPT_DIR}/quantize.py" \
    "${OUTPUT_DIR}/qwen2.5-3b-f16.gguf" \
    "${OUTPUT_DIR}/qwen2.5-3b-q4_0.gguf" \
    q4_0

# =============================================================================
# Convert Model 2: Qwen3-4B-Instruct-2507
# =============================================================================
echo "Converting Model 2: Qwen3-4B-Instruct-2507"

python "${SCRIPT_DIR}/convert.py" \
    "${MODEL_DIR_2}" \
    --outtype f16 \
    --outfile "${OUTPUT_DIR}/qwen3-4b-f16.gguf"

echo "(Quantizing to Q4...)"
python "${SCRIPT_DIR}/quantize.py" \
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