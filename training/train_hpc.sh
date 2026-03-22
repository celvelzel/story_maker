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

MODEL="microsoft/Phi-3-mini-4k-instruct"
DATASET="data/combined_data.jsonl"
OUTPUT_DIR="output/Phi-3_$(date +%Y%m%d_%H%M)"
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

# 可改参数: 模型路径
# 可改参数: 数据集路径（去掉 #100 表示使用全部数据）
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE swift sft \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --tuner_type lora \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --save_steps 100 \
    --logging_steps 5 \
    --save_total_limit 3 \
    --seed 42 \
    --report_to none

ELAPSED=$SECONDS
log "[4/4] 训练完成，耗时 ${ELAPSED}s"
echo "🎉 输出目录: $OUTPUT_DIR"