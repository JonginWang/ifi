# ============================================================================
# VEST Monitoring PowerShell Script
# ============================================================================

param(
    [switch]$Help
)

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptPath
Set-Location $ProjectRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

if ($Help) {
    Write-Host "VEST Monitoring PowerShell Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run_vest_monitoring.ps1 QUERY [OPTIONS]"
    Write-Host ""
    Write-Host "Query formats:" -ForegroundColor Yellow
    Write-Host "  Single shot: '47807'"
    Write-Host "  Space list : '47805 47807 47808'"
    Write-Host "  Range      : '47807:47840'"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_vest_monitoring.ps1 '47807'"
    Write-Host "  .\run_vest_monitoring.ps1 '47807:47840'"
    Write-Host "  .\run_vest_monitoring.ps1 '47805 47807 47808' --plot_each"
    Write-Host "  .\run_vest_monitoring.ps1 '47805 47807 47808' --plot_each --auto_close_sec 2"
    Write-Host "  .\run_vest_monitoring.ps1 '47807:47840' --xrange '0.28 0.35' --xcoil '1 5 6 10'"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --config PATH         Config file path (default: ifi/config.ini)"
    Write-Host "  --results_dir DIR     Results directory (default: ifi/results)"
    Write-Host "  --xrange 'A B'        Time range in seconds (default: '0.28 0.35')"
    Write-Host "  --xcoil 'LIST'        Coil IDs (default: '1 5 6 10')"
    Write-Host "  --gas H2|He           Gas selection (default: H2)"
    Write-Host "  --overwrite_local     Overwrite local monitoring outputs"
    Write-Host "  --no_save_plots       Skip PNG save"
    Write-Host "  --plot_each           Process plots shot-by-shot and save under each results/<shot>/monitoring"
    Write-Host "  --auto_close_sec N    With --plot_each, close each figure automatically after N seconds"
    Write-Host "  --help                Show this help message"
    Write-Host ""
    exit 0
}

if ($args.Count -lt 1) {
    Write-Host "ERROR: No query specified." -ForegroundColor Red
    & $MyInvocation.MyCommand.Path -Help
    exit 1
}

$query = $args[0]
$pythonArgs = @()
if ($args.Count -gt 1) {
    $pythonArgs = $args[1..($args.Count - 1)]
}

$Command = "python scripts/run_vest_monitoring.py `"$query`" $($pythonArgs -join ' ')"
Write-Host "Executing: $Command" -ForegroundColor Gray
Write-Host ""

Invoke-Expression $Command
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    exit $ExitCode
}
