# ==============================================================================
# PowerShell Wrapper for Smart Analysis Runner
# ==============================================================================
# This script wraps the Python smart analysis runner for convenient PowerShell usage.
#
# Usage:
#   .\run_analysis_smart.ps1 45000:45010 [additional args...]
#   .\run_analysis_smart.ps1 45000:45010 --freq 280 --density --stft --save_data
#
# Example:
#   .\run_analysis_smart.ps1 45000:45010 --freq 280 --density --stft --stft_cols 0 1 --save_plots --save_data --color_density_by_amplitude --vest_fields 102 109 101 144 214 171
# ==============================================================================

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ShotRange,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$AdditionalArgs
)

# Get script directory and project root
$scriptDir = $PSScriptRoot
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

# Activate virtual environment if it exists
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

# Build command arguments
$pythonScript = Join-Path $scriptDir "run_analysis_smart.py"
$allArgs = @($ShotRange) + $AdditionalArgs

# Run the Python script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Smart Analysis Runner" -ForegroundColor Green
Write-Host "Shot Range: $ShotRange" -ForegroundColor Yellow
Write-Host "Additional Args: $($AdditionalArgs -join ' ')" -ForegroundColor Gray
Write-Host "Working Directory: $projectRoot" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$process = Start-Process -FilePath "python" `
                         -ArgumentList @($pythonScript) + $allArgs `
                         -WorkingDirectory $projectRoot `
                         -Wait `
                         -NoNewWindow `
                         -PassThru

if ($process.ExitCode -ne 0) {
    Write-Host "[ERROR] Smart analysis failed with exit code: $($process.ExitCode)" -ForegroundColor Red
    exit $process.ExitCode
} else {
    Write-Host "[SUCCESS] Smart analysis completed successfully." -ForegroundColor Green
    exit 0
}

