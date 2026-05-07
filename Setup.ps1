# ── PowerShell Setup Script ──
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║   Zava University - On-Device AI Showcase  SETUP             ║" -ForegroundColor Blue
Write-Host "║   Powered by Microsoft Surface + Foundry Local              ║" -ForegroundColor Blue
Write-Host "║   Optimized for Snapdragon X NPU (QNN)                     ║" -ForegroundColor Blue
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

# Check for Snapdragon
try {
    $cpuName = (Get-CimInstance Win32_Processor).Name
    if ($cpuName -match "Qualcomm|Snapdragon") {
        Write-Host "[INFO] Snapdragon detected: $cpuName" -ForegroundColor Cyan
        Write-Host "       Ensure you're using ARM64-native Python for best NPU performance." -ForegroundColor Yellow
        Write-Host ""
    }
} catch {}

# Resolve Python command
$pythonCmd = $null
foreach ($cmd in @('python', 'py', 'python3')) {
    try {
        $testArgs = if ($cmd -eq 'py') { @('-3', '--version') } else { @('--version') }
        & $cmd @testArgs 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0 -or $?) {
            $pythonCmd = if ($cmd -eq 'py') { 'py -3' } else { $cmd }
            break
        }
    } catch { }
}
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    Write-Host "        For Snapdragon: install the ARM64 version for best performance." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python found: $pythonCmd" -ForegroundColor Green

# Check Foundry Local
$foundryOk = $false
try { foundry --version 2>&1 | Out-Null; $foundryOk = $true } catch {}
if (-not $foundryOk) {
    Write-Host "[INFO] Installing Foundry Local..." -ForegroundColor Yellow
    winget install Microsoft.FoundryLocal
}

# Create venv
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[SETUP] Creating virtual environment..." -ForegroundColor Cyan
    Invoke-Expression "$pythonCmd -m venv .venv"
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# Install deps
Write-Host "[SETUP] Installing dependencies..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "[OK] Setup complete! Run StartApp.bat or: python app.py" -ForegroundColor Green
Write-Host "     Then open http://localhost:5003" -ForegroundColor Green
Write-Host "     NOTE: Foundry Local model loading may take a moment on first run." -ForegroundColor Yellow
Write-Host ""
