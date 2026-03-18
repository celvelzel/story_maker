@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PORT=7860"
set "LOG_DIR=logs"
set "TS=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TS=%TS: =0%"
set "LOG_FILE=%LOG_DIR%\storyweaver_prod_%TS%.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo [INFO] StoryWeaver production launcher > "%LOG_FILE%"
echo [INFO] Timestamp: %date% %time% >> "%LOG_FILE%"

echo ===========================================
echo StoryWeaver Production Bootstrap (Windows)
echo ===========================================
echo [INFO] Log file: %LOG_FILE%

call :get_listen_pid %PORT% RUNNING_PID
if defined RUNNING_PID (
    call :is_storyweaver_process %RUNNING_PID% IS_SW
    if /I "!IS_SW!"=="1" (
        echo [INFO] Existing StoryWeaver PID !RUNNING_PID! detected on %PORT%. Restarting safely...
        echo [INFO] Existing StoryWeaver PID !RUNNING_PID! detected on %PORT%. Restarting safely...>> "%LOG_FILE%"
        taskkill /PID !RUNNING_PID! /F >> "%LOG_FILE%" 2>&1
        timeout /t 1 /nobreak >nul
    ) else (
        echo [ERROR] Port %PORT% is occupied by non-StoryWeaver PID !RUNNING_PID!.
        echo [ERROR] Port %PORT% is occupied by non-StoryWeaver PID !RUNNING_PID!.>> "%LOG_FILE%"
        exit /b 3
    )
)

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if %errorlevel%==0 (
        echo [STEP] Creating virtual environment with py...
        echo [STEP] Creating virtual environment with py...>> "%LOG_FILE%"
        py -3 -m venv .venv >> "%LOG_FILE%" 2>&1
    ) else (
        echo [STEP] Creating virtual environment with python...
        echo [STEP] Creating virtual environment with python...>> "%LOG_FILE%"
        python -m venv .venv >> "%LOG_FILE%" 2>&1
    )
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo [ERROR] Failed to create virtual environment.>> "%LOG_FILE%"
        exit /b 1
    )
)

set "PYTHON_CMD=.venv\Scripts\python.exe"

echo [STEP] Upgrading pip...
echo [STEP] Upgrading pip...>> "%LOG_FILE%"
"%PYTHON_CMD%" -m pip install --upgrade pip --timeout 60 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    echo [ERROR] Failed to upgrade pip.>> "%LOG_FILE%"
    exit /b 1
)

echo [STEP] Installing requirements with timeout controls...
echo [STEP] Installing requirements with timeout controls...>> "%LOG_FILE%"
"%PYTHON_CMD%" -m pip install -r requirements.txt --timeout 60 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo [ERROR] Dependency installation failed.>> "%LOG_FILE%"
    exit /b 1
)

echo [STEP] Launching Streamlit on port %PORT%...
echo [STEP] Launching Streamlit on port %PORT%...>> "%LOG_FILE%"
echo [INFO] Streamlit runs in foreground for production. This terminal will stay active.
echo [INFO] Open: http://127.0.0.1:%PORT%
echo [INFO] Full runtime logs: %LOG_FILE%
echo [INFO] Press Ctrl+C to stop the app.
echo [INFO] Streamlit runs in foreground for production. This terminal will stay active.>> "%LOG_FILE%"
echo [INFO] Open: http://127.0.0.1:%PORT%>> "%LOG_FILE%"
echo [INFO] Full runtime logs: %LOG_FILE%>> "%LOG_FILE%"
echo [INFO] Press Ctrl+C to stop the app.>> "%LOG_FILE%"
"%PYTHON_CMD%" -m streamlit run app.py --server.port=%PORT% --server.headless=true >> "%LOG_FILE%" 2>&1
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo [ERROR] App exited with code %APP_EXIT%. See %LOG_FILE%
    echo [ERROR] App exited with code %APP_EXIT%. See %LOG_FILE%>> "%LOG_FILE%"
    exit /b %APP_EXIT%
)

exit /b 0

:get_listen_pid
setlocal
set "PORT_TO_CHECK=%~1"
set "FOUND_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:":%PORT_TO_CHECK% .*LISTENING"') do (
    if not "%%P"=="0" set "FOUND_PID=%%P"
)
endlocal & set "%~2=%FOUND_PID%"
exit /b 0

:is_storyweaver_process
setlocal
set "PID=%~1"
set "IS_SW=0"
for /f "usebackq delims=" %%C in (`powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter 'ProcessId=%PID%' ^| Select-Object -ExpandProperty CommandLine)"`) do (
    echo %%C | findstr /I /C:"streamlit" /C:"app.py" >nul && set "IS_SW=1"
)
endlocal & set "%~2=%IS_SW%"
exit /b 0
