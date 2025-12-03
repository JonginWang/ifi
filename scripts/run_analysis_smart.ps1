# ==============================================================================
# PowerShell Wrapper for Smart Analysis Runner
# ==============================================================================
# This script wraps the Python smart analysis runner for convenient PowerShell usage.
# Supports multiple shot ranges via --shot-list parameter.
# Uses the same argument format as the batch file (--stft, --density, etc.)
#
# Usage:
#   Single range: .\run_analysis_smart.ps1 45000:45010 [additional args...]
#   Multiple ranges: .\run_analysis_smart.ps1 --shot-list "45000:45010" "45020:45030" [additional args...]
#   Example list: .\run_analysis_smart.ps1 [additional args...] (uses example list)
#
# Example:
#   .\run_analysis_smart.ps1 45000:45010 --freq 280 --density --stft --save_data
#   .\run_analysis_smart.ps1 --shot-list "45000:45010" "45020:45030" --freq 280 --density --stft --save_data
# ==============================================================================

param(
    [switch]$Help
)

# Get script directory and project root
$scriptDir = $PSScriptRoot
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $projectRoot

# Activate virtual environment if it exists
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

# Show help if requested
if ($Help) {
    Write-Host "Smart Analysis Runner - PowerShell Wrapper" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  Single range: .\run_analysis_smart.ps1 SHOT_RANGE [OPTIONS]"
    Write-Host "  Multiple ranges: .\run_analysis_smart.ps1 --shot-list 'RANGE1' 'RANGE2' [OPTIONS]"
    Write-Host "  Example list: .\run_analysis_smart.ps1 [OPTIONS] (uses example list)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_analysis_smart.ps1 '45000:45010' --freq '280' --density --stft --save_data"
    Write-Host "  .\run_analysis_smart.ps1 --shot-list '45000:45010' '45020:45030' --freq '94 280' --density --stft --save_data"
    Write-Host "  .\run_analysis_smart.ps1 --freq '94 280' --density --stft --save_data --stft-cols '0 1 2'"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --shot-list RANGES          Space-separated shot ranges (colon for ranges, e.g., '45000:45010' '45020:45030')"
    Write-Host "                             If not provided, uses example list from script"
    Write-Host "  --freq FREQ                 Filter frequencies (space-separated in quotes, e.g., '94 280')"
    Write-Host "  --stft-cols INDICES         Column indices for STFT (space-separated in quotes, e.g., '0 1 2')"
    Write-Host "  --cwt-cols INDICES          Column indices for CWT (space-separated in quotes, e.g., '0 1')"
    Write-Host "  All other arguments are passed directly to the Python script"
    Write-Host "  --help                     Show this help message"
    Write-Host ""
    Write-Host "Notes:" -ForegroundColor Yellow
    Write-Host "  - Arrays use space-separated values in quotes (default)"
    Write-Host "  - Ranges use colon separator (e.g., '45687:45689')"
    Write-Host "  - Comma-separated values are automatically converted to space-separated"
    Write-Host ""
    exit 0
}

# Get all arguments (PowerShell $args contains all positional arguments)
$allArgsList = $args

# Parse arguments to handle --shot-list and passthrough
$shotRanges = @()
$pythonArgs = @()
$i = 0
$singleRange = $null

# Process all arguments
while ($i -lt $allArgsList.Count) {
    $arg = $allArgsList[$i]
    
    if ($arg -eq "--shot-list" -or $arg -eq "--shot_list") {
        # Collect shot ranges until next -- option
        $i++
        while ($i -lt $allArgsList.Count -and -not $allArgsList[$i].StartsWith("--")) {
            $shotRanges += $allArgsList[$i]
            $i++
        }
        continue
    } elseif ($arg -eq "--help" -or $arg -eq "-help") {
        # Help already handled above
        $i++
        continue
    } elseif ($arg -eq "--freq" -or $arg -eq "--stft-cols" -or $arg -eq "--stft_cols" -or $arg -eq "--cwt-cols" -or $arg -eq "--cwt_cols") {
        # Handle options that accept comma-separated values
        $pythonArgs += $arg
        $i++
        if ($i -lt $allArgsList.Count) {
            $value = $allArgsList[$i]
            # If value contains comma, split and add as separate arguments
            if ($value -match ",") {
                $values = $value -split "," | ForEach-Object { $_.Trim() }
                $pythonArgs += $values
            } else {
                $pythonArgs += $value
            }
            $i++
        }
        continue
    } elseif (-not $arg.StartsWith("--") -and $null -eq $singleRange) {
        # First non-option argument is the shot range
        $singleRange = $arg
        $i++
        continue
    } else {
        # Passthrough to Python
        $pythonArgs += $arg
        $i++
    }
}

# Build command arguments for Python script
$pythonScript = Join-Path $scriptDir "run_analysis_smart.py"
$allArgs = @()

# Determine shot ranges
if ($shotRanges.Count -gt 0) {
    # Multiple ranges via --shot-list argument
    $allArgs += "--shot-list"
    $allArgs += $shotRanges
} elseif ($singleRange) {
    # Single range via positional parameter
    $allArgs += $singleRange
} else {
    # No shot range specified - will use example list
    Write-Host "No shot range specified. Using example shot list from script." -ForegroundColor Gray
    Write-Host ""
}

# Add all passthrough arguments
$allArgs += $pythonArgs

# Display information
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Smart Analysis Runner" -ForegroundColor Green
if ($shotRanges.Count -gt 0) {
    Write-Host "Shot Ranges: $($shotRanges -join ', ')" -ForegroundColor Yellow
} elseif ($singleRange) {
    Write-Host "Shot Range: $singleRange" -ForegroundColor Yellow
} else {
    Write-Host "Using example shot list" -ForegroundColor Yellow
}
if ($pythonArgs.Count -gt 0) {
    Write-Host "Additional Args: $($pythonArgs -join ' ')" -ForegroundColor Gray
}
Write-Host "Working Directory: $projectRoot" -ForegroundColor Gray
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Run the Python script
$process = Start-Process -FilePath "python" `
                         -ArgumentList (@($pythonScript) + $allArgs) `
                         -WorkingDirectory $projectRoot `
                         -Wait `
                         -NoNewWindow `
                         -PassThru

if ($process.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Smart analysis failed with exit code: $($process.ExitCode)" -ForegroundColor Red
    exit $process.ExitCode
} else {
    Write-Host ""
    Write-Host "[SUCCESS] Smart analysis completed successfully." -ForegroundColor Green
    exit 0
}
