#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
MODEL_PATH="$PROJECT_DIR/models/nlg/merged_model_Qwen3-4B-Instruct-2507/"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/inference_server_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found at $MODEL_PATH"
    exit 1
fi

echo "Starting local inference server..."
echo "Model: $MODEL_PATH"
echo "Log: $LOG_FILE"

"$VENV_DIR/bin/python" "$SCRIPT_DIR/local_inference_server.py" \
    --model_path "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$PROJECT_DIR/logs/inference_server.pid"
echo "Inference server started with PID: $SERVER_PID"