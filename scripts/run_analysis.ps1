# ============================================================================
# IFI Analysis PowerShell Script
# ============================================================================
# This PowerShell script provides convenient commands for running IFI analysis
# Uses the same argument format as the batch file (--stft, --density, etc.)
# ============================================================================

param(
    [switch]$Help
)

# Set project root directory
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptPath
Set-Location $ProjectRoot

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

# Show help if requested
if ($Help) {
    Write-Host "IFI Analysis PowerShell Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run_analysis.ps1 SHOT_NUMBER [OPTIONS]"
    Write-Host "  .\run_analysis.ps1 --query SHOT_NUMBER [OPTIONS]"
    Write-Host ""
    Write-Host "Query formats:" -ForegroundColor Yellow
    Write-Host "  Single shot: 45687 or --query 45687"
    Write-Host "  Range (colon): '45687:45689' or --query '45687:45689' (processes 45687, 45688, 45689)"
    Write-Host "  List (comma): '45687,45689' or --query '45687,45689' (processes 45687, 45689)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_analysis.ps1 45821"
    Write-Host "  .\run_analysis.ps1 --query 45821"
    Write-Host "  .\run_analysis.ps1 --query '45687:45689'"
    Write-Host "  .\run_analysis.ps1 --query '45687,45689'"
    Write-Host "  .\run_analysis.ps1 45687:45689"
    Write-Host "  .\run_analysis.ps1 45687,45689"
    Write-Host "  .\run_analysis.ps1 45821 --cwt"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --plot"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --scheduler processes"
    Write-Host "  .\run_analysis.ps1 45821:45825 --cwt"
    Write-Host "  .\run_analysis.ps1 45687:45679 --cwt"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --baseline ip"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --color-density-by-amplitude"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --color-density-by-amplitude --amplitude-colormap viridis"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --vest-fields 109 110"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --no-plot-raw --no-plot-ft"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --freq 94"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --freq '94,280'"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --freq '94 280'"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --stft-cols '0,1,2'"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --stft-cols 1"
    Write-Host "  .\run_analysis.ps1 45821 --cwt --stft-cols '0 1 2'"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --cwt                        Enable CWT analysis"
    Write-Host "  --plot                       Enable interactive plotting"
    Write-Host "  --no-plot                    Disable plotting (default)"
    Write-Host "  --scheduler TYPE             Dask scheduler: threads, processes, single-threaded"
    Write-Host "  --processes                  Use processes scheduler (alias)"
    Write-Host "  --single                     Use single-threaded scheduler (alias)"
    Write-Host "  --no-save                    Don't save results to HDF5"
    Write-Host "  --save-plots                 Save plots to files"
    Write-Host "  --force-remote               Force fetching from remote NAS"
    Write-Host "  --baseline TYPE              Baseline correction: ip or trig"
    Write-Host "  --data-folders FOLDERS       Comma-separated list of data folders"
    Write-Host "  --add-path                   Add data folders to default paths"
    Write-Host "  --results-dir DIR            Directory for results (default: ifi/results)"
    Write-Host "  --no-offset-removal          Disable offset removal"
    Write-Host "  --offset-window SIZE         Window size for offset removal (default: 2001)"
    Write-Host "  --stft-cols INDICES          Column indices for STFT (comma or space separated)"
    Write-Host "  --cwt-cols INDICES           Column indices for CWT (comma or space separated)"
    Write-Host "  --no-plot-block              Non-blocking plot mode"
    Write-Host "  --no-plot-raw                Don't plot raw data"
    Write-Host "  --no-plot-ft                 Don't plot time-frequency transforms"
    Write-Host "  --downsample FACTOR          Downsample factor for plotting (default: 10)"
    Write-Host "  --trigger-time SECONDS       Trigger time in seconds (default: 0.290)"
    Write-Host "  --vest-fields FIELDS         Space-separated VEST DB field IDs"
    Write-Host "  --color-density-by-amplitude Color-code density plots by amplitude"
    Write-Host "  --amplitude-colormap MAP     Colormap for amplitude (default: coolwarm)"
    Write-Host "  --amplitude-impedance OHMS   System impedance in ohms (default: 50.0)"
    Write-Host "  --freq FREQ                 Filter to specific frequency(ies): 94, 280"
    Write-Host "                              Can specify multiple: --freq 94 280"
    Write-Host "  --help                      Show this help message"
    Write-Host ""
    exit 0
}

# Get all arguments (PowerShell $args contains all positional arguments)
$allArgsList = $args

# Parse arguments to handle --query and passthrough
$query = $null
$pythonArgs = @()
$i = 0

# Process all arguments
while ($i -lt $allArgsList.Count) {
    $arg = $allArgsList[$i]
    
    if ($arg -eq "--query" -or $arg -eq "-query") {
        # Get query value
        $i++
        if ($i -lt $allArgsList.Count) {
            $query = $allArgsList[$i]
            $i++
        }
        continue
    } elseif ($arg -eq "--help" -or $arg -eq "-help") {
        # Help already handled above
        $i++
        continue
    } elseif (-not $arg.StartsWith("--") -and $null -eq $query) {
        # First non-option argument is the query
        $query = $arg
        $i++
        continue
    } else {
        # Passthrough to Python
        $pythonArgs += $arg
        $i++
    }
}

# Check if query is provided
if (-not $query) {
    Write-Host "ERROR: No query specified." -ForegroundColor Red
    Write-Host ""
    & $MyInvocation.MyCommand.Path -Help
    exit 1
}

# Build command arguments (default: --stft --density, same as bat file)
$ArgsList = @()
$ArgsList += "--stft"
$ArgsList += "--density"

# Add all passthrough arguments
$ArgsList += $pythonArgs

# Display information
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Running IFI Analysis" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Query: $query" -ForegroundColor Yellow
Write-Host "Options: $($ArgsList -join ' ')" -ForegroundColor Yellow
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Run analysis
$Command = "python -m ifi.analysis.main_analysis $query $($ArgsList -join ' ')"
Write-Host "Executing: $Command" -ForegroundColor Gray
Write-Host ""

Invoke-Expression $Command
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host "Analysis completed successfully!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host "Analysis failed with error code: $ExitCode" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Red
    exit $ExitCode
}
