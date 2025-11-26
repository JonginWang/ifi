@echo off
REM ============================================================================
REM IFI Data Save Batch Script
REM ============================================================================
REM This batch script provides convenient commands for saving IFI analysis data
REM ============================================================================
REM
REM Usage:
REM   run_datasave.bat SHOT_NUMBER [OPTIONS]
REM
REM Modes:
REM   --signal-only    Save only processed signals (no density calculation)
REM   --density        Save processed signals + density data (full analysis)
REM
REM Examples:
REM   run_datasave.bat 45821 --signal-only
REM   run_datasave.bat 45821 --density
REM   run_datasave.bat 45821:45825 --density --freq 94.0
REM   run_datasave.bat 45821 --density --freq 280.0
REM   run_datasave.bat 45821 --density --freq 94.0 280.0
REM ============================================================================

setlocal enabledelayedexpansion

REM Set project root directory
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Default values
set "SCHEDULER=threads"
set "MODE="
set "FREQ_DISPLAY="
set "ARGS="

REM Parse command line arguments
set "QUERY="

:parse_args
if "%~1"=="" goto :check_mode
if "%~1"=="--signal-only" (
    set "MODE=signal-only"
    shift
    goto :parse_args
)
if "%~1"=="--density" (
    set "MODE=density"
    shift
    goto :parse_args
)
if "%~1"=="--freq" (
    REM Start collecting frequency values for --freq
    set "FREQ_DISPLAY="
    set "ARGS=!ARGS! --freq"
    shift
    :collect_freq
    if "%~1"=="" goto :parse_args
    REM Check if next argument starts with -- (another option), if so stop collecting
    set "ARG_VAL=%~1"
    echo !ARG_VAL! | findstr /R "^--" >nul
    if !ERRORLEVEL! EQU 0 goto :parse_args
    REM Treat any non-option arguments after --freq as frequency values.
    REM Validation of allowed values (94.0, 280.0) is handled by Python argparse.
    set "ARGS=!ARGS! !ARG_VAL!"
    if "!FREQ_DISPLAY!"=="" (
        set "FREQ_DISPLAY=!ARG_VAL!"
    ) else (
        set "FREQ_DISPLAY=!FREQ_DISPLAY!, !ARG_VAL!"
    )
    shift
    goto :collect_freq
)
if "%~1"=="--scheduler" (
    set "ARGS=!ARGS! --scheduler %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--processes" (
    set "ARGS=!ARGS! --scheduler processes"
    shift
    goto :parse_args
)
if "%~1"=="--single" (
    set "ARGS=!ARGS! --scheduler single-threaded"
    shift
    goto :parse_args
)
if "%~1"=="--force-remote" (
    set "ARGS=!ARGS! --force_remote"
    shift
    goto :parse_args
)
if "%~1"=="--baseline" (
    set "ARGS=!ARGS! --baseline %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--vest-fields" (
    set "ARGS=!ARGS! --vest_fields %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--data-folders" (
    set "ARGS=!ARGS! --data_folders %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--add-path" (
    set "ARGS=!ARGS! --add_path"
    shift
    goto :parse_args
)
if "%~1"=="--results-dir" (
    set "ARGS=!ARGS! --results_dir %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-offset-removal" (
    set "ARGS=!ARGS! --no_offset_removal"
    shift
    goto :parse_args
)
if "%~1"=="--offset-window" (
    set "ARGS=!ARGS! --offset_window %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--stft-cols" (
    set "ARGS=!ARGS! --stft_cols %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--cwt-cols" (
    set "ARGS=!ARGS! --cwt_cols %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--downsample" (
    set "ARGS=!ARGS! --downsample %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--trigger-time" (
    set "ARGS=!ARGS! --trigger_time %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--color-density-by-amplitude" (
    set "ARGS=!ARGS! --color_density_by_amplitude"
    shift
    goto :parse_args
)
if "%~1"=="--amplitude-colormap" (
    set "ARGS=!ARGS! --amplitude_colormap %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--amplitude-impedance" (
    set "ARGS=!ARGS! --amplitude_impedance %~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_help
)
if "!QUERY!"=="" (
    set "QUERY=%~1"
) else (
    REM After the first positional argument (QUERY), treat all others as passthrough args.
    set "ARGS=!ARGS! %~1"
)
shift
goto :parse_args

:check_mode
if "!QUERY!"=="" (
    echo ERROR: No query specified.
    echo.
    goto :show_help
)
if "!MODE!"=="" (
    echo ERROR: No mode specified. Use --signal-only or --density
    echo.
    goto :show_help
)

echo ============================================================================
echo Running IFI Data Save
echo ============================================================================
echo Query: !QUERY!
echo Mode: !MODE!
if not "!FREQ_DISPLAY!"=="" (
    echo Frequency filter: !FREQ_DISPLAY!
)
echo Scheduler: !SCHEDULER!
echo Additional options: !ARGS!
echo ============================================================================
echo.

if "!MODE!"=="signal-only" (
    REM Signal only: STFT analysis, no density, save data
    python -m ifi.analysis.main_analysis !QUERY! --stft --save_data --scheduler !SCHEDULER! !ARGS!
) else if "!MODE!"=="density" (
    REM Full analysis: STFT + density, save data
    python -m ifi.analysis.main_analysis !QUERY! --stft --density --save_data --scheduler !SCHEDULER! !ARGS!
) else (
    echo ERROR: Unknown mode: !MODE!
    exit /b 1
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo Data save completed successfully!
    echo ============================================================================
) else (
    echo.
    echo ============================================================================
    echo Data save failed with error code: %ERRORLEVEL%
    echo ============================================================================
    exit /b %ERRORLEVEL%
)

goto :end

:show_help
echo IFI Data Save Batch Script
echo.
echo Usage:
echo   run_datasave.bat SHOT_NUMBER [MODE] [OPTIONS]
echo.
echo Modes (required):
echo   --signal-only         Save only processed signals (no density calculation)
echo   --density             Save processed signals + density data (full analysis)
echo.
echo Examples:
echo   run_datasave.bat 45821 --signal-only
echo   run_datasave.bat 45821 --density
echo   run_datasave.bat 45821:45825 --density
echo   run_datasave.bat 45821 --density --freq 94
echo   run_datasave.bat 45821 --density --freq 280
echo   run_datasave.bat 45821 --density --freq 94 280
echo   run_datasave.bat 45821 --density --freq 94.0 280.0
echo   run_datasave.bat 45821 --density --baseline ip
echo   run_datasave.bat 45821 --density --vest-fields 109 110
echo.
echo Options:
echo   --freq FREQ                  Filter to specific frequency(ies): 94, 280, 94.0, 280.0
echo                                Can specify multiple: --freq 94 280
echo   --scheduler TYPE             Dask scheduler: threads, processes, single-threaded
echo   --processes                  Use processes scheduler (alias)
echo   --single                     Use single-threaded scheduler (alias)
echo   --force-remote               Force fetching from remote NAS
echo   --baseline TYPE              Baseline correction: ip or trig
echo   --vest-fields FIELDS         Space-separated VEST DB field IDs
echo   --data-folders FOLDERS       Comma-separated list of data folders
echo   --add-path                   Add data folders to default paths
echo   --results-dir DIR             Directory for results (default: ifi/results)
echo   --no-offset-removal          Disable offset removal
echo   --offset-window SIZE         Window size for offset removal (default: 2001)
echo   --stft-cols INDICES          Space-separated column indices for STFT
echo   --cwt-cols INDICES           Space-separated column indices for CWT
echo   --downsample FACTOR          Downsample factor for plotting (default: 10)
echo   --trigger-time SECONDS       Trigger time in seconds (default: 0.290)
echo   --color-density-by-amplitude Color-code density plots by amplitude
echo   --amplitude-colormap MAP     Colormap for amplitude (default: coolwarm)
echo   --amplitude-impedance OHMS   System impedance in ohms (default: 50.0)
echo   --help                       Show this help message
echo.
echo Notes:
echo   - Signal-only mode: Processes and saves signals without density calculation
echo   - Density mode: Full analysis including phase and density calculation
echo   - Frequency filter: Process only specified frequencies (94GHz or 280GHz)
echo   - If --freq is not specified, all available frequencies are processed
echo.
goto :end

:end
endlocal

