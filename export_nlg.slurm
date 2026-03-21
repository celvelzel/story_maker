#!/bin/bash
# ==============================================================================
# export_nlg.slurm
# 合并 LoRA 权重 + 导出 GGUF 量化模型
# 步骤 1: swift export --merge_lora true    (LoRA -> 完整模型)
# 步骤 2: swift export --to_gguf true       (完整模型 -> GGUF Q8_0)
# 平台: PolyU Student HPC (SLURM)
# ==============================================================================
#SBATCH --job-name=export_nlg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err

# ── 环境加载 ───────────────────────────────────────────────────
module purge
module load anaconda3
module load cuda/12.1
source activate swift_env

# ── 变量配置 ───────────────────────────────────────────────────
MODEL_TYPE="llama3_2-3b-instruct"
# 训练输出的 checkpoint 目录 (根据实际训练结果修改 checkpoint 编号)
CHECKPOINT="/puhome/2511696g/NLP/output/nlg_model/checkpoint-100"
MERGED_DIR="/puhome/2511696g/NLP/output/nlg_model_merged"
GGUF_DIR="/puhome/2511696g/NLP/output/nlg_model_gguf"
HF_HOME="/puhome/2511696g/NLP/hf_cache"

mkdir -p "$MERGED_DIR" "$GGUF_DIR" logs
export HF_HOME="$HF_HOME"

echo "=========================================="
echo "StoryWeaver NLG Model Export"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Merged dir: $MERGED_DIR"
echo "GGUF dir:   $GGUF_DIR"
echo "Date:       $(date)"
echo "=========================================="

# ── 检查 checkpoint ────────────────────────────────────────────
if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -d /puhome/2511696g/NLP/output/nlg_model/checkpoint-* 2>/dev/null || echo "  (none found)"
    exit 1
fi

# ── 步骤 1: 合并 LoRA 权重 ─────────────────────────────────────
echo ""
echo ">>> Step 1: Merging LoRA weights..."
swift export \
    --model_type "$MODEL_TYPE" \
    --adapters "$CHECKPOINT" \
    --merge_lora true \
    --output_dir "$MERGED_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: LoRA merge failed"
    exit 1
fi
echo "Merged model saved to: $MERGED_DIR"

# ── 步骤 2: 导出 GGUF (Q8_0 量化) ─────────────────────────────
echo ""
echo ">>> Step 2: Exporting to GGUF (Q8_0 quantization)..."
swift export \
    --model_type "$MODEL_TYPE" \
    --model_id_or_path "$MERGED_DIR" \
    --to_gguf true \
    --quant_bits 8 \
    --quant_method q8_0 \
    --output_dir "$GGUF_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: GGUF export failed"
    exit 1
fi

# ── 完成 ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Export complete!"
echo "=========================================="
echo ""
echo "GGUF files:"
ls -lh "$GGUF_DIR"/*.gguf 2>/dev/null || echo "  (check output directory)"
echo ""
echo "Next steps:"
echo "  1. Download .gguf to local machine:"
echo "     scp 2511696g@hpc.polyu.edu.hk:$GGUF_DIR/*.gguf ./"
echo "  2. Create Modelfile:"
echo "     echo 'FROM ./llama3.2-3b-instruct-q8_0.gguf' > Modelfile"
echo "  3. Import to Ollama:"
echo "     ollama create storyweaver-llama3 -f Modelfile"
echo "  4. Run: ollama serve"
echo "  5. Update .env:"
echo "     OPENAI_BASE_URL=http://localhost:11434/v1"
echo "     OPENAI_MODEL=storyweaver-llama3"
