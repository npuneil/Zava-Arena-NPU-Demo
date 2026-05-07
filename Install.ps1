<#
.SYNOPSIS
    Zava University On-Device AI Showcase - Full Installer
    Installs Python venv, Foundry Local, downloads models, starts service.
.NOTES
    This script uses only ASCII characters to avoid encoding issues.
#>

param(
    [switch]$SkipFoundryInstall,
    [switch]$SkipModelDownload,
    [string]$PreferredDevice = "Auto"
)

$ErrorActionPreference = 'Continue'
$Script:TotalSteps = 6
$Script:FoundryInstalled = $false
$Script:FoundryVersion = ""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
function Write-Banner {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Blue
    Write-Host "  Zava University - On-Device AI Showcase  INSTALLER" -ForegroundColor White
    Write-Host "  Powered by Microsoft Surface + Foundry Local" -ForegroundColor Gray
    Write-Host "  Optimized for Snapdragon X NPU (QNN)" -ForegroundColor Gray
    Write-Host "================================================================" -ForegroundColor Blue
    Write-Host ""
}

function Write-Step {
    param([int]$Step, [int]$Total, [string]$Message)
    Write-Host "[$Step/$Total] $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "  [WARN] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "  [ERROR] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "  [INFO] $Message" -ForegroundColor Gray
}

# ---------------------------------------------------------------------------
# STEP 0: Banner and environment detection
# ---------------------------------------------------------------------------
Write-Banner

$cpuName = ""
$isSnapdragon = $false
$isIntel = $false
try {
    $cpuName = (Get-CimInstance Win32_Processor).Name
    if ($cpuName -match "Qualcomm|Snapdragon") {
        $isSnapdragon = $true
        Write-Ok "Snapdragon detected: $cpuName"
    } elseif ($cpuName -match "Intel") {
        $isIntel = $true
        Write-Ok "Intel detected: $cpuName"
    } else {
        Write-Ok "CPU detected: $cpuName"
    }
} catch {
    Write-Warn "Could not detect CPU"
}

# ---------------------------------------------------------------------------
# STEP 1: Python
# ---------------------------------------------------------------------------
Write-Step 1 $Script:TotalSteps "Checking Python"

$pythonCmd = $null
foreach ($cmd in @('python', 'py', 'python3')) {
    try {
        $testArgs = if ($cmd -eq 'py') { @('-3', '--version') } else { @('--version') }
        $result = & $cmd @testArgs 2>&1
        if ($LASTEXITCODE -eq 0 -or $?) {
            $pythonCmd = if ($cmd -eq 'py') { 'py -3' } else { $cmd }
            $pyVersion = ($result | Out-String).Trim()
            break
        }
    } catch { }
}
if (-not $pythonCmd) {
    Write-Err "Python not found. Install Python 3.10+ from https://python.org"
    if ($isSnapdragon) {
        Write-Err "For Snapdragon: install the ARM64 version for best performance."
    }
    exit 1
}
Write-Ok "Python found: $pythonCmd ($pyVersion)"

# ---------------------------------------------------------------------------
# STEP 2: Virtual environment
# ---------------------------------------------------------------------------
Write-Step 2 $Script:TotalSteps "Setting up Python virtual environment"

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Info "Creating virtual environment..."
    Invoke-Expression "$pythonCmd -m venv .venv"
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
        Write-Err "Failed to create virtual environment."
        exit 1
    }
    Write-Ok "Virtual environment created"
} else {
    Write-Ok "Virtual environment already exists"
}

# Activate and install deps
Write-Info "Installing dependencies..."
& .\.venv\Scripts\Activate.ps1
& .\.venv\Scripts\pip.exe install --quiet -r requirements.txt
Write-Ok "Dependencies installed"

# ---------------------------------------------------------------------------
# STEP 3: Foundry Local
# ---------------------------------------------------------------------------
Write-Step 3 $Script:TotalSteps "Checking Foundry Local"

# Check if already installed
try {
    $fv = foundry --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $Script:FoundryInstalled = $true
        $Script:FoundryVersion = ($fv | Out-String).Trim()
        Write-Ok "Foundry Local installed: $($Script:FoundryVersion)"
    }
} catch {}

if (-not $Script:FoundryInstalled -and -not $SkipFoundryInstall) {
    Write-Info "Foundry Local not found. Attempting install via winget..."
    try {
        $wingetResult = winget install Microsoft.FoundryLocal --accept-package-agreements --accept-source-agreements 2>&1
        $wingetOutput = $wingetResult | Out-String

        if ($wingetOutput -match "Successfully installed" -or $wingetOutput -match "already installed") {
            # Refresh PATH
            $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
            $env:PATH = "$machinePath;$userPath"

            # Verify installation
            $fv2 = foundry --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $Script:FoundryInstalled = $true
                $Script:FoundryVersion = ($fv2 | Out-String).Trim()
                Write-Ok "Foundry Local installed: $($Script:FoundryVersion)"
            } else {
                Write-Warn "Foundry installed but not in PATH. Restart terminal and re-run."
            }
        } else {
            Write-Warn "winget install did not confirm success."
            Write-Info "Install manually: winget install Microsoft.FoundryLocal"
        }
    } catch {
        Write-Warn "Could not auto-install Foundry Local."
        Write-Info "Install manually: winget install Microsoft.FoundryLocal"
    }
} elseif (-not $Script:FoundryInstalled) {
    Write-Warn "Foundry Local not found (install skipped with -SkipFoundryInstall)"
    Write-Info "Install manually: winget install Microsoft.FoundryLocal"
}

# ---------------------------------------------------------------------------
# STEP 4: Start Foundry Service and verify models
# ---------------------------------------------------------------------------
Write-Step 4 $Script:TotalSteps "Starting Foundry Service and Verifying Models"

if (-not $Script:FoundryInstalled) {
    Write-Warn "Skipping - Foundry Local not installed"
} else {
    # Start the service
    Write-Info "Starting Foundry Local service..."
    try {
        $statusOutput = foundry service status 2>&1 | Out-String
        if ($statusOutput -match "http://") {
            Write-Ok "Foundry service already running"
        } else {
            foundry service start 2>&1 | Out-Null
            Start-Sleep -Seconds 3
            Write-Ok "Foundry service started"
        }
    } catch {
        Write-Warn "Could not start Foundry service"
    }

    # List available models
    Write-Info "Checking available models..."
    try {
        $modelOutput = foundry model list 2>&1 | Out-String
        $npuModels = @()
        $cpuModels = @()

        foreach ($line in ($modelOutput -split "`n")) {
            $trimmed = $line.Trim()
            # Look for model aliases with device info
            if ($trimmed -match '^\s*(\S+)\s+(NPU|QNN|CPU|GPU)') {
                $alias = $Matches[1]
                $device = $Matches[2]
                if ($device -eq "NPU" -or $device -eq "QNN") {
                    $npuModels += $alias
                } elseif ($device -eq "CPU") {
                    $cpuModels += $alias
                }
            }
            # Also match model variant lines
            elseif ($trimmed -match '(\S+-(?:cpu|npu|qnn)\S*)') {
                $variant = $Matches[1]
                if ($variant -match "npu|qnn") {
                    $npuModels += $variant
                } elseif ($variant -match "cpu") {
                    $cpuModels += $variant
                }
            }
        }

        if ($npuModels.Count -gt 0) {
            Write-Ok "NPU models available: $($npuModels -join ', ')"
        }
        if ($cpuModels.Count -gt 0) {
            Write-Ok "CPU models available: $($cpuModels -join ', ')"
        }
        if ($npuModels.Count -eq 0 -and $cpuModels.Count -eq 0) {
            Write-Info "No cached models found. Models will download on first run."
        }
    } catch {
        Write-Warn "Could not list models"
    }
}

# ---------------------------------------------------------------------------
# STEP 5: Model download (optional)
# ---------------------------------------------------------------------------
Write-Step 5 $Script:TotalSteps "Model Pre-download (optional)"

if (-not $Script:FoundryInstalled -or $SkipModelDownload) {
    Write-Info "Skipping model pre-download"
} else {
    # Build list of models to offer
    $modelsToLoad = @()

    if ($isSnapdragon) {
        # Prefer QNN/NPU models for Snapdragon
        $npuModel = "qwen2.5-1.5b"
        $modelsToLoad += @{ Alias = $npuModel; Label = "qwen2.5-1.5b NPU (QNN) - 2.78 GB"; Device = "NPU" }
    }

    if ($isIntel) {
        # Prefer OpenVINO NPU models for Intel
        $npuModel = "phi-4-mini"
        $modelsToLoad += @{ Alias = $npuModel; Label = "phi-4-mini NPU (OpenVINO) - ~3 GB"; Device = "NPU" }
    }

    # CPU fallback model - works on all platforms
    $cpuModel = "Phi-4-mini-instruct-generic-cpu"
    $modelsToLoad += @{ Alias = $cpuModel; Label = "Phi-4-mini CPU - 4.80 GB"; Device = "CPU" }

    if ($modelsToLoad.Count -gt 0) {
        Write-Info "Recommended models for your hardware:"
        $i = 1
        foreach ($m in $modelsToLoad) {
            Write-Host "    $i. $($m.Label)" -ForegroundColor White
            $i++
        }
        Write-Host ""
        Write-Info "Models will auto-download on first use if not pre-loaded."
        Write-Info "To pre-download now, run: foundry model load <model-alias>"
    }
}

# ---------------------------------------------------------------------------
# STEP 6: Final summary
# ---------------------------------------------------------------------------
Write-Step 6 $Script:TotalSteps "Installation Complete"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Python:      $pythonCmd ($pyVersion)" -ForegroundColor White
Write-Host "  Venv:        .venv (activated)" -ForegroundColor White
if ($Script:FoundryInstalled) {
    Write-Host "  Foundry:     $($Script:FoundryVersion)" -ForegroundColor White
} else {
    Write-Host "  Foundry:     NOT INSTALLED" -ForegroundColor Yellow
}
Write-Host "  CPU:         $cpuName" -ForegroundColor White
Write-Host ""
Write-Host "  To start the demo:" -ForegroundColor Cyan
Write-Host "    1. Run: StartApp.bat" -ForegroundColor White
Write-Host "    2. Open: http://localhost:5003" -ForegroundColor White
Write-Host ""
Write-Host "  First run may take a few minutes to download the AI model." -ForegroundColor Yellow
Write-Host ""
