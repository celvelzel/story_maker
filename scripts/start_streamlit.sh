#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/streamlit_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

echo "Starting Streamlit frontend..."
echo "Log: $LOG_FILE"

"$VENV_DIR/bin/streamlit" run "$PROJECT_DIR/app.py" \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > "$LOG_FILE" 2>&1 &

STREAMLIT_PID=$!
echo $STREAMLIT_PID > "$PROJECT_DIR/logs/streamlit.pid"
echo "Streamlit started with PID: $STREAMLIT_PID"