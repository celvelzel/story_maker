@echo off
setlocal enabledelayedexpansion

REM ================================================================
REM  StoryWeaver Local Deployment - KoboldCpp Vulkan + Streamlit
REM  本地部署脚本：AMD 780M Vulkan 加速 + Streamlit 前端
REM ================================================================

cd /d "%~dp0.."

set "KOBOLDCPP=C:\Tools\KoboldCpp\koboldcpp.exe"
set "MODEL_PATH=models\qwen-gguf\qwen3-4b-q4_k_m.gguf"
set "LLM_PORT=5001"
set "APP_PORT=7860"
set "GPULAYERS=99"
set "CONTEXT_SIZE=2048"
set "THREADS=8"

set "LOG_DIR=logs"
set "TS=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TS=%TS: =0%"
set "LOG_FILE=%LOG_DIR%\vulkan_deploy_%TS%.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo [INFO] Vulkan Deployment Log > "%LOG_FILE%"
echo [INFO] Timestamp: %date% %time% >> "%LOG_FILE%"

echo ===========================================
echo  StoryWeaver - Local Vulkan Deployment
echo ===========================================
echo.

REM ── Step 0: Kill existing processes on LLM port ──────────────
echo [STEP] Checking for existing processes on port %LLM_PORT%...
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:":%LLM_PORT% .*LISTENING"') do (
    if not "%%P"=="0" (
        echo [INFO] Killing existing process PID %%P on port %LLM_PORT%...
        taskkill /PID %%P /F >nul 2>&1
    )
)
timeout /t 1 /nobreak >nul

REM ── Step 1: Verify KoboldCpp binary ─────────────────────────
echo [STEP] Verifying KoboldCpp binary...
if not exist "%KOBOLDCPP%" (
    echo [ERROR] KoboldCpp not found at: %KOBOLDCPP%
    echo [HINT] Download from: https://github.com/LostRuins/koboldcpp/releases
    echo [HINT] Use the nocuda build for Vulkan support.
    pause
    exit /b 1
)
echo [OK]   KoboldCpp found.

REM ── Step 2: Verify model file ───────────────────────────────
echo [STEP] Verifying model file...
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model not found at: %MODEL_PATH%
    echo [INFO] Available models:
    dir /b models\qwen-gguf\ 2>nul || echo   (directory empty)
    pause
    exit /b 1
)
echo [OK]   Model found: %MODEL_PATH%

REM ── Step 3: Verify virtual environment ──────────────────────
echo [STEP] Checking Python environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo [HINT] Run: python -m venv .venv
    echo [HINT] Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK]   Python environment found.

REM ── Step 4: Start KoboldCpp with Vulkan ─────────────────────
echo.
echo [STEP] Starting KoboldCpp with Vulkan acceleration...
echo [INFO] Model:      %MODEL_PATH%
echo [INFO] GPU Layers: %GPULAYERS%
echo [INFO] Context:    %CONTEXT_SIZE% tokens
echo [INFO] Threads:    %THREADS%
echo [INFO] LLM API:    http://127.0.0.1:%LLM_PORT%/v1
echo.

start "KoboldCpp-Vulkan" /B "%KOBOLDCPP%" ^
    --usevulkan ^
    --gpulayers %GPULAYERS% ^
    --model "%MODEL_PATH%" ^
    --contextsize %CONTEXT_SIZE% ^
    --threads %THREADS% ^
    --port %LLM_PORT% ^
    >> "%LOG_FILE%" 2>&1

set "KOBOLDCPP_PID=%ERRORLEVEL%"

REM ── Step 5: Wait for LLM server to be ready ─────────────────
echo [STEP] Waiting for LLM server to be ready...
set "MAX_WAIT=120"
set "WAITED=0"
:wait_loop
    timeout /t 3 /nobreak >nul
    set /a WAITED+=3
    powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%LLM_PORT%/v1/models' -UseBasicParsing -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
    if %errorlevel%==0 (
        echo [OK]   LLM server ready after %WAITED%s.
        goto :llm_ready
    )
    if %WAITED% GEQ %MAX_WAIT% (
        echo [ERROR] LLM server did not start within %MAX_WAIT%s.
        echo [HINT] Check log: %LOG_FILE%
        pause
        exit /b 1
    )
    echo [WAIT] Still waiting... (%WAITED%s / %MAX_WAIT%s)
    goto :wait_loop

:llm_ready

REM ── Step 6: Verify API is working ───────────────────────────
echo [STEP] Verifying LLM API...
powershell -Command "$r = Invoke-WebRequest -Uri 'http://127.0.0.1:%LLM_PORT%/v1/models' -UseBasicParsing; $r.Content" >> "%LOG_FILE%" 2>&1
echo [OK]   LLM API is responding.

REM ── Step 7: Start Streamlit app ─────────────────────────────
echo.
echo [STEP] Starting StoryWeaver Streamlit app...
echo [INFO] App URL:   http://127.0.0.1:%APP_PORT%
echo [INFO] LLM API:   http://127.0.0.1:%LLM_PORT%/v1
echo.
echo ===========================================
echo  Services Running:
echo    LLM  : http://127.0.0.1:%LLM_PORT%  (KoboldCpp + Vulkan)
echo    App  : http://127.0.0.1:%APP_PORT%  (Streamlit)
echo ===========================================
echo.
echo  Press Ctrl+C to stop.
echo ===========================================
echo.

.venv\Scripts\python.exe -m streamlit run app.py ^
    --server.port=%APP_PORT% ^
    --server.headless=true

echo.
echo [INFO] Services stopped.
pause
