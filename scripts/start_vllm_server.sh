#!/usr/bin/env bash
# start_vllm_server.sh - Start vLLM OpenAI-compatible API server
#
# This script starts the vLLM server for serving the fine-tuned Qwen model
# with an OpenAI-compatible API endpoint.
#
# Usage: ./scripts/start_vllm_server.sh

set -euo pipefail

# --- Load environment variables ---
ENV_FILE=".env.vllm"

if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Warning: $ENV_FILE not found, using default values"
fi

# --- Configuration (with defaults) ---
MODEL_PATH="${MODEL_PATH:?ERROR: MODEL_PATH must be set}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-local}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_API_KEY="${VLLM_API_KEY:-}"

echo "=== vLLM Server Configuration ==="
echo "  Model:          $MODEL_PATH"
echo "  Served as:      $SERVED_MODEL_NAME"
echo "  Host:           $VLLM_HOST"
echo "  Port:           $VLLM_PORT"
echo "  Max Model Len:  $MAX_MODEL_LEN"
echo "  GPU Memory:     $GPU_MEMORY_UTILIZATION"
echo "================================="
echo ""

# --- Build command ---
CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --trust-remote-code
)

# Only add api-key if it's set
if [[ -n "$VLLM_API_KEY" ]]; then
    CMD+=(--api-key "$VLLM_API_KEY")
fi

# --- Start server (foreground) ---
echo "Starting vLLM server..."
exec "${CMD[@]}"
