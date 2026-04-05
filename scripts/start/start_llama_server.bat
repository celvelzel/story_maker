@echo off
setlocal enabledelayedexpansion

REM llama.cpp Local API Server Launcher
REM 启动本地 llama-server (OpenAI 兼容 API)
REM 使用 Qwen3-4B 量化模型，适合 CPU 推理

cd /d "%~dp0.."

set "LLAMA_BIN=llama.cpp-bin\llama-server.exe"
set "MODEL_PATH=models\qwen-gguf\qwen3-4b-q4_k_m.gguf"
set "PORT=8081"
set "HOST=127.0.0.1"
set "CONTEXT_SIZE=2048"
set "BATCH_SIZE=512"
set "THREADS=4"

echo ==========================================
echo  llama.cpp Local API Server
echo ==========================================
echo.

REM 检查 llama-server 可执行文件
if not exist "%LLAMA_BIN%" (
    echo [ERROR] llama-server.exe not found at: %LLAMA_BIN%
    echo Please download llama.cpp binaries from:
    echo https://github.com/ggerganov/llama.cpp/releases
    echo and extract to llama.cpp-bin/
    pause
    exit /b 1
)

REM 检查模型文件
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model file not found at: %MODEL_PATH%
    echo.
    echo Available models in models\qwen-gguf\:
    dir /b models\qwen-gguf\ 2>nul || echo   (directory empty or not found)
    echo.
    echo Please run the conversion pipeline first:
    echo   1. scripts\convert_to_gguf.bat
    echo   2. llama-quantize (see docs)
    pause
    exit /b 1
)

echo [INFO] Model:    %MODEL_PATH%
echo [INFO] Server:   http://%HOST%:%PORT%
echo [INFO] Context:  %CONTEXT_SIZE% tokens
echo [INFO] Threads:  %THREADS%
echo.
echo OpenAI-compatible endpoints:
echo   Chat:    http://%HOST%:%PORT%/v1/chat/completions
echo   Models:  http://%HOST%:%PORT%/v1/models
echo.
echo Press Ctrl+C to stop the server.
echo ==========================================
echo.

REM 启动 llama-server
"%LLAMA_BIN%" ^
    --model "%MODEL_PATH%" ^
    --host %HOST% ^
    --port %PORT% ^
    --ctx-size %CONTEXT_SIZE% ^
    --batch-size %BATCH_SIZE% ^
    --threads %THREADS% ^
    --chat-template chatml

echo.
echo [INFO] Server stopped.
pause
