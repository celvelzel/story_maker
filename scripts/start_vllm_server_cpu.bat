@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0.."

REM Find Git Bash
for %%i in ("C:\Program Files\Git\bin\bash.exe" "C:\Program Files (x86)\Git\bin\bash.exe") do (
    if exist "%%i" (
        "%%i" --login ./scripts/start_vllm_server_cpu.sh
        exit /b !ERRORLEVEL!
    )
)

echo Error: Git Bash not found at expected locations
echo Please ensure Git for Windows is installed
pause
exit /b 1
