$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Test-PythonCommand {
    param(
        [string]$Command,
        [string[]]$Args
    )

    try {
        & $Command @Args -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Invoke-SetupWizard {
    param(
        [string]$Command,
        [string[]]$Args
    )

    & $Command @Args "setup_wizard.py"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Gemini Lovart Image Automation - Windows setup"
Write-Host "================================================"

if (Test-PythonCommand -Command "py" -Args @("-3.12")) {
    Invoke-SetupWizard -Command "py" -Args @("-3.12")
}

if (Test-PythonCommand -Command "py" -Args @("-3")) {
    Invoke-SetupWizard -Command "py" -Args @("-3")
}

if (Test-PythonCommand -Command "python" -Args @()) {
    Invoke-SetupWizard -Command "python" -Args @()
}

if (Test-PythonCommand -Command "python3" -Args @()) {
    Invoke-SetupWizard -Command "python3" -Args @()
}

Write-Host ""
Write-Host "Python 3.12 or newer was not found."
Write-Host "Please install Python first, then run this setup again."
Write-Host ""
Write-Host "Recommended Windows command:"
Write-Host "  winget install -e --id Python.Python.3.12"
Write-Host ""
Write-Host "Or download it from:"
Write-Host "  https://www.python.org/downloads/windows/"
Write-Host ""
Write-Host "After installation, close this terminal, open a new one, and run setup_windows.bat again."
exit 1
