#!/bin/bash
set -Eeuo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

on_error() {
    local exit_code=$?
    local line_no=${BASH_LINENO[0]}
    local failed_cmd="${BASH_COMMAND}"

    log "[ERROR] 训练失败"
    log "[ERROR] exit_code=$exit_code"
    log "[ERROR] line=$line_no"
    log "[ERROR] command=$failed_cmd"
    exit "$exit_code"
}

trap 'on_error' ERR

# 1. 设置 Hugging Face Access Token
export HF_TOKEN="your_huggingface_access_token"  # 替换为你的 HF token

MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET="data/combined_data.jsonl"
OUTPUT_DIR="output/Llama-3.2-3B-Instruct_$(date +%Y%m%d_%H%M)"
CUDA_DEVICE="0"

log "[1/4] 初始化环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate SHPC-env

log "[2/4] 环境已激活: SHPC-env"
log "训练配置: model=$MODEL, dataset=$DATASET, output=$OUTPUT_DIR, gpu=$CUDA_DEVICE"

if [ ! -f "$DATASET" ]; then
    log "错误: 找不到数据集文件 $DATASET"
    exit 1
fi

log "[3/4] 开始训练（每 5 step 打印一次日志，100 step 保存一次 checkpoint）..."
SECONDS=0

# 2. 传入 HF token（通过 --hf_token 参数）
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE swift sft \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --tuner_type lora \
    --output_dir "$OUTPUT_DIR" \
    --use_hf true \
    --hf_token "$HF_TOKEN" \  # 传入 access token
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --save_steps 100 \
    --logging_steps 5 \
    --save_total_limit 2 \
    --seed 42 \
    --report_to none

ELAPSED=$SECONDS
log "[4/4] 训练完成，耗时 ${ELAPSED}s"
echo "🎉 输出目录: $OUTPUT_DIR"
