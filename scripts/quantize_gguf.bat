@echo off
REM GGUF Quantization Script - 将 FP16 模型量化為 INT4 (Q4)
REM 使用 llama.cpp Windows 預編譯版

setlocal enabledelayedexpansion

cd /d "%~dp0.."

echo ========================================
echo  GGUF INT4 Quantization
echo ========================================
echo.

REM 设置路径
set INPUT_FILE=.\models\qwen-gguf\qwen3-4b-f16.gguf
set OUTPUT_FILE=.\models\qwen-gguf\qwen3-4b-q4_0.gguf
set LLAMA_DIR=.\llama.cpp-bin

REM 检查输入文件是否存在
if not exist "%INPUT_FILE%" (
    echo Error: Input file not found: %INPUT_FILE%
    echo Please run convert_to_gguf.bat first!
    pause
    exit /b 1
)

REM 下载预编译版 llama.cpp (如果没有)
if not exist "%LLAMA_DIR%\llama-quantize.exe" (
    echo Downloading llama.cpp binaries...
    curl -L -o llama-bin.zip https://github.com/ggerganov/llama.cpp/releases/download/b3807/llama.cpp-bin-b3807-windows-x64.zip
    powershell -Command "Expand-Archive -Path llama-bin.zip -DestinationPath . -Force"
    move llama.cpp-bin-b3807-windows-x64 "%LLAMA_DIR%" 2>nul
    del llama-bin.zip
    echo Download complete!
)

REM 运行量化
echo.
echo Quantizing to INT4 (Q4_0)...
echo Input:  %INPUT_FILE%
echo Output: %OUTPUT_FILE%
echo This may take 5-10 minutes...
echo.

"%LLAMA_DIR%\llama-quantize.exe" ^
    "%INPUT_FILE%" ^
    "%OUTPUT_FILE%" ^
    q4_0

echo.
echo ========================================
echo Quantization Complete!
echo ========================================
echo Output: %OUTPUT_DIR%\qwen3-4b-q4_0.gguf
echo.

REM 显示文件大小
powershell -Command "Get-Item '%OUTPUT_FILE%' | Select-Object Name, @{Name='Size(MB)';Expression={[math]::Round($_.Length/1MB,2)}}"

pause