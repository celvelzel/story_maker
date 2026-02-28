@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM ── Existing instance detection ───────────────────────────
set "PORT=7860"
set "FALLBACK_PORT=7861"
set "FORCE_RESTART=0"

if /I "%~1"=="--force-restart" set "FORCE_RESTART=1"
if /I "%~1"=="-f" set "FORCE_RESTART=1"
if not "%~1"=="" (
    if /I not "%~1"=="--force-restart" (
        if /I not "%~1"=="-f" (
            echo [ERROR] Unknown argument: %~1
            echo [INFO] Usage: start_project.bat [--force-restart ^| -f]
            exit /b 1
        )
    )
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do set "RUNNING_7860=%%a"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7861') do set "RUNNING_7861=%%a"

if "%FORCE_RESTART%"=="1" (
    if defined RUNNING_7860 (
        echo [INFO] --force-restart enabled. Stopping PID %RUNNING_7860% on 7860...
        taskkill /PID %RUNNING_7860% /F >nul 2>nul
    )
    if defined RUNNING_7861 (
        echo [INFO] --force-restart enabled. Stopping PID %RUNNING_7861% on 7861...
        taskkill /PID %RUNNING_7861% /F >nul 2>nul
    )
    timeout /t 1 /nobreak >nul
    set "RUNNING_7860="
    set "RUNNING_7861="
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do set "RUNNING_7860=%%a"
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7861') do set "RUNNING_7861=%%a"
    if defined RUNNING_7860 (
        echo [ERROR] Failed to stop existing PID %RUNNING_7860% on 7860.
        exit /b 1
    )
    if defined RUNNING_7861 (
        echo [ERROR] Failed to stop existing PID %RUNNING_7861% on 7861.
        exit /b 1
    )
)

if defined RUNNING_7860 (
    echo [INFO] StoryWeaver appears to be already running on http://127.0.0.1:7860, PID %RUNNING_7860%.
    echo [INFO] Reusing existing instance. No new process started.
    exit /b 0
)

if defined RUNNING_7861 (
    echo [INFO] StoryWeaver appears to be already running on http://127.0.0.1:7861, PID %RUNNING_7861%.
    echo [INFO] Reusing existing instance. No new process started.
    exit /b 0
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT%') do set "OCCUPIED_PID=%%a"
if defined OCCUPIED_PID (
    echo [WARN] Port %PORT% is occupied by PID %OCCUPIED_PID%. Using fallback port %FALLBACK_PORT%.
    set "PORT=%FALLBACK_PORT%"
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT%') do set "FALLBACK_OCCUPIED_PID=%%a"
if defined FALLBACK_OCCUPIED_PID (
    echo [ERROR] Both 7860 and 7861 are occupied. Stop old instances and retry.
    exit /b 1
)

echo ==========================================
echo StoryWeaver Bootstrap ^& Launch (Windows)
echo ==========================================
if "%FORCE_RESTART%"=="1" echo [INFO] Mode: force restart
echo [INFO] Will launch on port: %PORT%
echo.

set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        echo [1/6] Creating virtual environment with py...
        py -3 -m venv .venv
        if errorlevel 1 goto :venv_fail
    ) else (
        where python >nul 2>nul
        if %errorlevel%==0 (
            echo [1/6] Creating virtual environment with python...
            python -m venv .venv
            if errorlevel 1 goto :venv_fail
        ) else (
            echo [ERROR] Python not found. Please install Python 3.10+ first.
            goto :fail
        )
    )
    set "PYTHON_CMD=.venv\Scripts\python.exe"
)

echo [2/6] Upgrading pip...
"%PYTHON_CMD%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/6] Installing requirements.txt...
"%PYTHON_CMD%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [4/6] Applying runtime compatibility fixes (gradio^<6)...
"%PYTHON_CMD%" -m pip install "gradio<6"
if errorlevel 1 goto :fail

echo [5/6] Downloading spaCy model en_core_web_sm (will continue if network times out)...
"%PYTHON_CMD%" -m spacy download en_core_web_sm
if errorlevel 1 (
    echo [WARN] spaCy model download failed. App can still start with fallback entity extraction.
)

if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] .env not found. Creating from .env.example...
        copy /y ".env.example" ".env" >nul
        echo [INFO] Please edit .env and set OPENAI_API_KEY if needed.
    ) else (
        echo [WARN] .env and .env.example are both missing. Create .env manually if API calls fail.
    )
)

echo [6/6] Launching app on http://127.0.0.1:%PORT% ...
set "GRADIO_SERVER_PORT=%PORT%"
"%PYTHON_CMD%" app.py
goto :end

:venv_fail
echo [ERROR] Failed to create virtual environment.
goto :fail

:fail
echo.
echo Bootstrapping failed. Please check the error message above.
exit /b 1

:end
endlocal
