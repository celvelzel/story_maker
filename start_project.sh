#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT=7860
FALLBACK_PORT=7861
FORCE_RESTART=0

if [[ "${1:-}" == "--force-restart" || "${1:-}" == "-f" ]]; then
  FORCE_RESTART=1
elif [[ -n "${1:-}" ]]; then
  echo "[ERROR] Unknown argument: $1"
  echo "[INFO] Usage: ./start_project.sh [--force-restart | -f]"
  exit 1
fi

get_pid_on_port() {
  local port="$1"
  lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
}

RUNNING_7860="$(get_pid_on_port 7860)"
RUNNING_7861="$(get_pid_on_port 7861)"

if [[ "$FORCE_RESTART" == "1" ]]; then
  if [[ -n "$RUNNING_7860" ]]; then
    echo "[INFO] --force-restart enabled. Stopping PID $RUNNING_7860 on 7860..."
    kill -9 "$RUNNING_7860" || true
  fi
  if [[ -n "$RUNNING_7861" ]]; then
    echo "[INFO] --force-restart enabled. Stopping PID $RUNNING_7861 on 7861..."
    kill -9 "$RUNNING_7861" || true
  fi
  sleep 1
  RUNNING_7860="$(get_pid_on_port 7860)"
  RUNNING_7861="$(get_pid_on_port 7861)"
  if [[ -n "$RUNNING_7860" ]]; then
    echo "[ERROR] Failed to stop existing PID $RUNNING_7860 on 7860."
    exit 1
  fi
  if [[ -n "$RUNNING_7861" ]]; then
    echo "[ERROR] Failed to stop existing PID $RUNNING_7861 on 7861."
    exit 1
  fi
fi

if [[ -n "$RUNNING_7860" ]]; then
  echo "[INFO] StoryWeaver appears to be already running on http://127.0.0.1:7860, PID $RUNNING_7860."
  echo "[INFO] Reusing existing instance. No new process started."
  exit 0
fi

if [[ -n "$RUNNING_7861" ]]; then
  echo "[INFO] StoryWeaver appears to be already running on http://127.0.0.1:7861, PID $RUNNING_7861."
  echo "[INFO] Reusing existing instance. No new process started."
  exit 0
fi

if [[ -n "$(get_pid_on_port "$PORT")" ]]; then
  echo "[WARN] Port $PORT is occupied. Using fallback port $FALLBACK_PORT."
  PORT="$FALLBACK_PORT"
fi

if [[ -n "$(get_pid_on_port "$PORT")" ]]; then
  echo "[ERROR] Both 7860 and 7861 are occupied. Stop old instances and retry."
  exit 1
fi

echo "=========================================="
echo "StoryWeaver Bootstrap & Launch (macOS/Linux)"
echo "=========================================="
if [[ "$FORCE_RESTART" == "1" ]]; then
  echo "[INFO] Mode: force restart"
fi
echo "[INFO] Will launch on port: $PORT"
echo

PYTHON_CMD=""
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  if command -v python3 >/dev/null 2>&1; then
    echo "[1/6] Creating virtual environment with python3..."
    python3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    echo "[1/6] Creating virtual environment with python..."
    python -m venv .venv
  else
    echo "[ERROR] Python not found. Please install Python 3.10+ first."
    exit 1
  fi
  PYTHON_CMD=".venv/bin/python"
fi

echo "[2/6] Upgrading pip..."
"$PYTHON_CMD" -m pip install --upgrade pip

echo "[3/6] Installing requirements.txt..."
"$PYTHON_CMD" -m pip install -r requirements.txt

echo "[4/6] Installing streamlit..."
"$PYTHON_CMD" -m pip install "streamlit>=1.30.0"

echo "[5/6] Downloading spaCy model en_core_web_sm (will continue if network times out)..."
if ! "$PYTHON_CMD" -m spacy download en_core_web_sm; then
  echo "[WARN] spaCy model download failed. App can still start with fallback entity extraction."
fi

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    echo "[INFO] .env not found. Creating from .env.example..."
    cp .env.example .env
    echo "[INFO] Please edit .env and set OPENAI_API_KEY if needed."
  else
    echo "[WARN] .env and .env.example are both missing. Create .env manually if API calls fail."
  fi
fi

echo "[6/6] Launching app on http://127.0.0.1:$PORT ..."
"$PYTHON_CMD" -m streamlit run app.py --server.port="$PORT" --server.headless=true
