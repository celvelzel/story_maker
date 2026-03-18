#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${STREAMLIT_PORT:-7860}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/storyweaver_prod_$(date +%Y%m%d_%H%M%S).log"

echo "===========================================" | tee -a "$LOG_FILE"
echo "StoryWeaver Production Bootstrap (macOS)" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
echo "[INFO] Log file: $LOG_FILE" | tee -a "$LOG_FILE"

validate_env() {
  if [[ -f ".env" ]]; then
    set +u
    # shellcheck disable=SC1091
    source .env
    set -u
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "[ERROR] OPENAI_API_KEY is missing in .env" | tee -a "$LOG_FILE"
      exit 2
    fi
    if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
      echo "[WARN] OPENAI_BASE_URL missing in .env, default may be used." | tee -a "$LOG_FILE"
    fi
    if [[ -z "${OPENAI_MODEL:-}" ]]; then
      echo "[WARN] OPENAI_MODEL missing in .env, default may be used." | tee -a "$LOG_FILE"
    fi
  else
    echo "[WARN] .env missing. Continuing with environment defaults." | tee -a "$LOG_FILE"
  fi
}

get_pid_on_port() {
  local port="$1"
  lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
}

is_storyweaver_process() {
  local pid="$1"
  local cmdline
  cmdline="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$cmdline" == *"streamlit"* && "$cmdline" == *"app.py"* ]]; then
    return 0
  fi
  return 1
}

validate_env

running_pid="$(get_pid_on_port "$PORT")"
if [[ -n "$running_pid" ]]; then
  if is_storyweaver_process "$running_pid"; then
    echo "[INFO] Existing StoryWeaver PID $running_pid detected on $PORT. Restarting safely..." | tee -a "$LOG_FILE"
    kill "$running_pid" || true
    sleep 1
  else
    echo "[ERROR] Port $PORT is occupied by non-StoryWeaver PID $running_pid." | tee -a "$LOG_FILE"
    exit 3
  fi
fi

if [[ ! -x ".venv/bin/python" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    echo "[STEP] Creating virtual environment with python3..." | tee -a "$LOG_FILE"
    python3 -m venv .venv >> "$LOG_FILE" 2>&1
  elif command -v python >/dev/null 2>&1; then
    echo "[STEP] Creating virtual environment with python..." | tee -a "$LOG_FILE"
    python -m venv .venv >> "$LOG_FILE" 2>&1
  else
    echo "[ERROR] Python not found." | tee -a "$LOG_FILE"
    exit 1
  fi
fi

PYTHON_CMD=".venv/bin/python"

echo "[STEP] Upgrading pip..." | tee -a "$LOG_FILE"
"$PYTHON_CMD" -m pip install --upgrade pip --timeout 60 >> "$LOG_FILE" 2>&1

echo "[STEP] Installing requirements with timeout controls..." | tee -a "$LOG_FILE"
"$PYTHON_CMD" -m pip install -r requirements.txt --timeout 60 >> "$LOG_FILE" 2>&1

echo "[STEP] Launching Streamlit on port $PORT..." | tee -a "$LOG_FILE"
set +e
"$PYTHON_CMD" -m streamlit run app.py --server.port="$PORT" --server.headless=true >> "$LOG_FILE" 2>&1
app_exit=$?
set -e

if [[ $app_exit -ne 0 ]]; then
  echo "[ERROR] App exited with code $app_exit. See $LOG_FILE" | tee -a "$LOG_FILE"
  exit "$app_exit"
fi

exit 0
