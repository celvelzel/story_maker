@echo off
REM GGUF Model Converter - Convert Qwen models to GGUF format for llama.cpp
REM Converts TWO models: qwen_2.5_3B + Qwen3-4B-Instruct-2507

setlocal enabledelayedexpansion

cd /d "%~dp0.."

echo ========================================
echo  Qwen to GGUF Converter (Batch)
echo ========================================
echo.

REM Set paths
set MODEL_DIR_1=.\models\nlg\qwen_2.5_3B
set MODEL_DIR_2=.\models\nlg\Qwen3-4B-Instruct-2507
set OUTPUT_DIR=.\models\qwen-gguf
set LLAMA_CPP_DIR=.\llama.cpp

REM Create output directory
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

REM Check if llama.cpp exists
if not exist "%LLAMA_CPP_DIR%" (
    echo Cloning llama.cpp repository...
    echo This may take a few minutes on first run...
    git clone https://github.com/ggerganov/llama.cpp.git "%LLAMA_CPP_DIR%"
    if errorlevel 1 (
        echo Failed to clone llama.cpp
        pause
        exit /b 1
    )
)

REM Install dependencies
echo Installing Python dependencies...
pip install -q numpy torch

echo.
echo ========================================
echo Converting Model 1: qwen_2.5_3B
echo ========================================
echo This may take 10-30 minutes...
echo.

python "%LLAMA_CPP_DIR%/convert.py" ^
    "%MODEL_DIR_1%" ^
    --outtype f16 ^
    --outfile "%OUTPUT_DIR%/qwen2.5-3b-f16.gguf"

echo.
echo ========================================
echo Converting Model 2: Qwen3-4B-Instruct-2507
echo ========================================
echo This may take 10-30 minutes...
echo.

python "%LLAMA_CPP_DIR%/convert.py" ^
    "%MODEL_DIR_2%" ^
    --outtype f16 ^
    --outfile "%OUTPUT_DIR%/qwen3-4b-f16.gguf"

echo.
echo ========================================
echo Conversion Complete!
echo ========================================
echo Outputs:
echo   1. %OUTPUT_DIR%\qwen2.5-3b-f16.gguf
echo   2. %OUTPUT_DIR%\qwen3-4b-f16.gguf
echo.
echo Next steps:
echo 1. Download llama.cpp binaries for Windows
echo 2. Quantize: llama-quantize %OUTPUT_DIR%\qwen2.5-3b-f16.gguf q4_0
echo 3. Run: llama-cli -m %OUTPUT_DIR%\qwen2.5-3b-f16.gguf -c 2048 -n 512
echo.

pause