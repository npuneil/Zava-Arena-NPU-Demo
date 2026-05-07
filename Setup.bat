@echo off
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   Zava University - On-Device AI Showcase  SETUP             ║
echo ║   Powered by Microsoft Surface + Foundry Local              ║
echo ║   Optimized for Snapdragon X NPU (QNN)                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM ── Check ARM64 vs x64 emulation ──
powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" 2>nul | findstr /i "qualcomm snapdragon" >nul
if not errorlevel 1 (
    echo [INFO] Snapdragon detected — ensure you're using ARM64-native Python for best NPU performance.
    echo        x64 emulated Python works but is slower and may have quirks.
    echo.
)

REM ── Resolve Python command ──
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_ok
)
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3
    goto :python_ok
)
python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :python_ok
)
echo [ERROR] Python is not installed or not in PATH.
echo         Install Python 3.10+ from https://python.org
echo         Make sure to check "Add Python to PATH" during install.
echo         For Snapdragon: install the ARM64 version for best performance.
pause
exit /b 1

:python_ok
echo [OK] Python found: %PYTHON_CMD%

REM ── Check for Foundry Local ──
foundry --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Foundry Local CLI not found. Installing...
    winget install Microsoft.FoundryLocal
    echo [INFO] Please restart this script after Foundry Local installs.
    pause
    exit /b 0
) else (
    echo [OK] Foundry Local detected.
)

REM ── Create virtual environment ──
if not exist ".venv\Scripts\activate.bat" (
    echo [SETUP] Creating Python virtual environment...
    %PYTHON_CMD% -m venv .venv
    if not exist ".venv\Scripts\activate.bat" (
        echo [ERROR] Failed to create virtual environment.
        echo         Try manually: %PYTHON_CMD% -m venv .venv
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM ── Activate and install dependencies ──
echo [SETUP] Installing Python dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [WARN] Some dependencies may have failed to install.
    echo        Try: pip install -r requirements.txt
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   Setup complete!                                           ║
echo ║   Run StartApp.bat to launch the demo.                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
pause
