@echo off
REM GGUF Model Converter - Convert Qwen model to GGUF format for llama.cpp
REM This script downloads llama.cpp and converts your model

setlocal enabledelayedexpansion

cd /d "%~dp0.."

echo ========================================
echo  Qwen to GGUF Converter
echo ========================================
echo.

REM Set paths
set MODEL_DIR=C:\Develop\python_projects\COMP5423_NLP\story_maker\models\nlg\qwen_2.5_3B
set OUTPUT_DIR=.\models\qwen-gguf
set LLAMA_CPP_DIR=.\llama.cpp

REM Create output directory
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

REM Check if llama.cpp exists
if not exist "%LLAMA_CPP_DIR%" (
    echo Cloning llama.cpp repository...
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

REM Run conversion
echo.
echo Converting model to GGUF format...
echo This may take 10-30 minutes depending on model size...
echo.

python "%LLAMA_CPP_DIR%/convert.py" ^
    "%MODEL_DIR%" ^
    --outtype f16 ^
    --outfile "%OUTPUT_DIR%/qwen2.5-3b-f16.gguf"

echo.
echo ========================================
echo Conversion complete!
echo Output: %OUTPUT_DIR%\qwen2.5-3b-f16.gguf
echo ========================================
echo.
echo Next steps:
echo 1. Download llama.cpp binaries for Windows
echo 2. Run: llama-cli -m %OUTPUT_DIR%\qwen2.5-3b-f16.gguf -c 2048 -n 512
echo.

pause