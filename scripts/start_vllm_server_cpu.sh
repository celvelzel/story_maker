#!/usr/bin/env bash
# start_vllm_server_cpu.sh - Start vLLM OpenAI-compatible API server (CPU-Only optimized)
#
# This script starts the vLLM server for serving the fine-tuned Qwen model
# on CPU-only hardware with INT4 quantization for efficient inference.
#
# Hardware Requirements:
#   - CPU: 8-core (e.g., AMD Ryzen 7 H255W)
#   - RAM: 32GB minimum
#   - Storage: Sufficient for quantized model (3B INT4 ≈ 2-3GB)
#
# Performance Note:
#   CPU-only inference is significantly slower than GPU inference.
#   Expected latency: 5-20 seconds per inference turn (depends on CPU and context length).
#   This configuration prioritizes memory efficiency and compatibility over speed.
#
# Usage: ./scripts/start_vllm_server_cpu.sh

set -euo pipefail

# --- Load environment variables ---
ENV_FILE=".env.vllm.cpu"

if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: $ENV_FILE not found. Please create it from .env.vllm.cpu.example or copy .env.vllm as .env.vllm.cpu"
    exit 1
fi

# --- Configuration (with defaults) ---
MODEL_PATH="${MODEL_PATH:?ERROR: MODEL_PATH must be set}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-local}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-int4}"
VLLM_CPU_ONLY="${VLLM_CPU_ONLY:-true}"
VLLM_MAX_PARALLEL_LOADING_WORKERS="${VLLM_MAX_PARALLEL_LOADING_WORKERS:-1}"

echo "========================================="
echo "  vLLM CPU-Only Server Configuration"
echo "========================================="
echo "  Model:                $MODEL_PATH"
echo "  Served as:            $SERVED_MODEL_NAME"
echo "  Host:                 $VLLM_HOST"
echo "  Port:                 $VLLM_PORT"
echo "  Max Model Len:        $MAX_MODEL_LEN"
echo "  Quantization:         $VLLM_QUANTIZATION"
echo "  CPU-Only Mode:        $VLLM_CPU_ONLY"
echo "  Parallel Workers:     $VLLM_MAX_PARALLEL_LOADING_WORKERS"
echo "========================================="
echo ""
echo "⚠️  NOTE: CPU-only inference is SLOW (5-20s per turn)"
echo "    Optimized for memory efficiency, not speed."
echo ""

# --- Build command ---
CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --quantization "$VLLM_QUANTIZATION"
    --cpu-offload-gb 4
    --max-parallel-loading-workers "$VLLM_MAX_PARALLEL_LOADING_WORKERS"
    --trust-remote-code
    --disable-log-requests
)

# Only add api-key if it's set
if [[ -n "$VLLM_API_KEY" ]]; then
    CMD+=(--api-key "$VLLM_API_KEY")
fi

# --- Start server (foreground) ---
echo "Starting vLLM CPU-Only server..."
echo "API endpoint: http://$VLLM_HOST:$VLLM_PORT/v1"
echo ""
exec "${CMD[@]}"
