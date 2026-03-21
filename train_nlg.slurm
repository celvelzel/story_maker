#!/bin/bash
# ==============================================================================
# train_nlg.slurm
# StoryWeaver NLG 模块微调脚本 — Llama-3.2-3B-Instruct + LoRA
# 平台: PolyU Student HPC (SLURM)
# ==============================================================================
#SBATCH --job-name=swift_nlg          # 任务名称
#SBATCH --partition=gpu               # GPU 队列 (若 HPC 分区名不同请修改)
#SBATCH --gres=gpu:1                  # 申请 1 块 GPU
#SBATCH --cpus-per-task=4             # CPU 核心数
#SBATCH --mem=32G                     # 内存
#SBATCH --time=04:00:00               # 最大运行时间 (4h, 777 条数据足够)
#SBATCH --output=logs/train_%j.out    # 标准输出日志
#SBATCH --error=logs/train_%j.err     # 错误日志

# ── 环境加载 ───────────────────────────────────────────────────
module purge
module load anaconda3
module load cuda/12.1                 # 根据 HPC 实际版本调整: module avail cuda

# ── Conda 环境 ─────────────────────────────────────────────────
source activate swift_env             # 若环境名不同请修改

# ── 变量配置 ───────────────────────────────────────────────────
MODEL_TYPE="llama3_2-3b-instruct"
DATASET="/puhome/2511696g/NLP/data/combined_data.jsonl"
OUTPUT_DIR="/puhome/2511696g/NLP/output/nlg_model"
HF_HOME="/puhome/2511696g/NLP/hf_cache"            # HuggingFace 缓存目录 (避免占用 home 配额)
HF_TOKEN="${HF_TOKEN:-}"                            # 从环境变量读取 HF token

# 创建必要目录
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir -p "$HF_HOME"
mkdir -p logs

# 设置 HuggingFace 缓存路径
export HF_HOME="$HF_HOME"

# ── 打印信息 ───────────────────────────────────────────────────
echo "=========================================="
echo "StoryWeaver NLG Fine-tuning"
echo "=========================================="
echo "Model:     $MODEL_TYPE"
echo "Dataset:   $DATASET"
echo "Output:    $OUTPUT_DIR"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Date:      $(date)"
echo "=========================================="

# ── 数据集检查 ─────────────────────────────────────────────────
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    exit 1
fi
SAMPLE_COUNT=$(wc -l < "$DATASET")
echo "Dataset samples: $SAMPLE_COUNT"

# ── 训练 ───────────────────────────────────────────────────────
swift sft \
    --model_type "$MODEL_TYPE" \
    --dataset "$DATASET" \
    --sft_type lora \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj v_proj \
    --max_length 2048 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --seed 42 \
    --deepspeed default-zero2

# ── 完成 ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Training complete!"
echo "Output: $OUTPUT_DIR"
echo "Date:   $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check output: ls $OUTPUT_DIR"
echo "  2. Merge LoRA:   swift export --model_type $MODEL_TYPE --adapters $OUTPUT_DIR/checkpoint-XXX --merge_lora true"
echo "  3. Export GGUF:  swift export --model_type $MODEL_TYPE --model_id_or_path <merged> --to_gguf true --quant_bits 8"
echo "  4. Download .gguf to local machine"
echo "  5. ollama create storyweaver -f Modelfile"
