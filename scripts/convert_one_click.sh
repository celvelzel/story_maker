#!/bin/bash
# 一键转换脚本 - 直接运行，无需配置

# =============================================================================
# 配置（根据你的路径）
# =============================================================================
CONVERT_SCRIPT="/puhome/25116696g/NLP/llama.cpp/convert_hf_to_gguf.py"
MODEL_DIR="/puhome/25116696g/NLP/merged_model_Qwen3-4B-Instruct-2507"
OUTPUT_FILE="/puhome/25116696g/NLP/qwen-gguf/qwen3-4b-f16.gguf"

# =============================================================================

echo "开始转换..."
echo "模型: ${MODEL_DIR}"
echo "输出: ${OUTPUT_FILE}"
echo ""

# 创建输出目录
mkdir -p /puhome/25116696g/NLP/qwen-gguf

# 安装依赖
pip install -q numpy torch transformers sentencepiece

# 运行转换
python "${CONVERT_SCRIPT}" \
    "${MODEL_DIR}" \
    --outtype f16 \
    --outfile "${OUTPUT_FILE}"

echo ""
echo "完成！"
echo "文件位置: ${OUTPUT_FILE}"
ls -lh "${OUTPUT_FILE}"