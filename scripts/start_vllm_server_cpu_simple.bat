@echo off
setlocal enabledelayedexpansion

REM Windows-Compatible vLLM CPU Server Launcher
REM Uses Python uvloop monkey-patch to work around Windows incompatibility

cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install vLLM
echo Installing vLLM...
pip install -q --upgrade pip 2>nul
pip install -q vllm 2>nul

REM Run Python launcher with uvloop monkey-patch
echo.
echo Starting vLLM server...
python scripts/start_vllm_server_cpu_windows.py

pause
