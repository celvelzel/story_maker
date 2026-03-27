@echo off
setlocal enabledelayedexpansion

REM Simple Windows batch launcher for vLLM CPU server
REM No Git Bash dependency required

cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install vLLM and all dependencies
echo Installing vLLM and dependencies...
pip install -q --upgrade pip
pip install -q vllm uvloop

REM Start vLLM server
echo.
echo Starting vLLM server on http://localhost:8000/v1
echo Model: @models\nlg\qwen_2.5_3B
echo Press Ctrl+C to stop
echo.

python -m vllm.entrypoints.openai.api_server ^
    --model @models\nlg\qwen_2.5_3B ^
    --quantization int4 ^
    --cpu-offload-gb 10 ^
    --tensor-parallel-size 1 ^
    --max-model-len 2048 ^
    --port 8000

pause
