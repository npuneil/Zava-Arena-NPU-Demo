@echo off
echo ╔══════════════════════════════════════════════════════════════╗
echo ║   Zava University - On-Device AI Showcase                    ║
echo ║   Starting...                                               ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM ── Activate venv ──
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found. Run Setup.bat first.
    pause
    exit /b 1
)

REM ── Launch browser after short delay ──
start "" "http://localhost:5003"

REM ── Start Flask app ──
echo [Starting] Flask app on http://localhost:5003
echo [INFO] Foundry Local model loading may take a moment on first run...
echo [INFO] The browser page will work once the app finishes loading.
echo [INFO] Close this window to stop the app.
echo.
python app.py
