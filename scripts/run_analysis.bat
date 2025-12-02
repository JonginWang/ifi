@echo off
REM ============================================================================
REM IFI Analysis Batch Script
REM ============================================================================
REM This batch script provides convenient commands for running IFI analysis
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
set "SAVE_DATA=--save_data"
set "PLOT="
set "CWT="

REM Parse command line arguments
set "QUERY="
set "ARGS="

:parse_args
if "%~1"=="" goto :run_analysis
if "%~1"=="--query" (
    set "QUERY=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--cwt" (
    set "CWT=--cwt"
    shift
    goto :parse_args
)
if "%~1"=="--plot" (
    set "PLOT=--plot"
    shift
    goto :parse_args
)
if "%~1"=="--no-plot" (
    set "PLOT="
    shift
    goto :parse_args
)
if "%~1"=="--scheduler" (
    set "SCHEDULER=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--processes" (
    set "SCHEDULER=processes"
    shift
    goto :parse_args
)
if "%~1"=="--single" (
    set "SCHEDULER=single-threaded"
    shift
    goto :parse_args
)
if "%~1"=="--no-save" (
    set "SAVE_DATA="
    shift
    goto :parse_args
)
if "%~1"=="--save-plots" (
    set "ARGS=!ARGS! --save_plots"
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
    REM Collect stft-cols values (can be comma or space separated)
    set "STFT_COLS_VAL="
    shift
    :collect_stft_cols
    if "%~1"=="" goto :apply_stft_cols
    REM Check if next argument starts with -- (another option), if so stop collecting
    set "ARG_VAL=%~1"
    echo !ARG_VAL! | findstr /R "^--" >nul
    if !ERRORLEVEL! EQU 0 goto :apply_stft_cols
    REM Treat any non-option arguments after --stft-cols as column indices
    if "!STFT_COLS_VAL!"=="" (
        set "STFT_COLS_VAL=!ARG_VAL!"
    ) else (
        REM If contains comma, it's comma-separated; otherwise space-separated
        echo !STFT_COLS_VAL! | findstr /C:"," >nul
        if !ERRORLEVEL! EQU 0 (
            REM Already comma-separated, add with comma
            set "STFT_COLS_VAL=!STFT_COLS_VAL!,!ARG_VAL!"
        ) else (
            REM Space-separated, add with space
            set "STFT_COLS_VAL=!STFT_COLS_VAL! !ARG_VAL!"
        )
    )
    shift
    goto :collect_stft_cols
    :apply_stft_cols
    REM Convert comma-separated to space-separated for Python argparse
    set "STFT_COLS_CONVERTED=!STFT_COLS_VAL!"
    set "STFT_COLS_CONVERTED=!STFT_COLS_CONVERTED:,= !"
    set "ARGS=!ARGS! --stft_cols !STFT_COLS_CONVERTED!"
    goto :parse_args
)
if "%~1"=="--cwt-cols" (
    REM Collect cwt-cols values (can be comma or space separated)
    set "CWT_COLS_VAL="
    shift
    :collect_cwt_cols
    if "%~1"=="" goto :apply_cwt_cols
    REM Check if next argument starts with -- (another option), if so stop collecting
    set "ARG_VAL=%~1"
    echo !ARG_VAL! | findstr /R "^--" >nul
    if !ERRORLEVEL! EQU 0 goto :apply_cwt_cols
    REM Treat any non-option arguments after --cwt-cols as column indices
    if "!CWT_COLS_VAL!"=="" (
        set "CWT_COLS_VAL=!ARG_VAL!"
    ) else (
        REM If contains comma, it's comma-separated; otherwise space-separated
        echo !CWT_COLS_VAL! | findstr /C:"," >nul
        if !ERRORLEVEL! EQU 0 (
            REM Already comma-separated, add with comma
            set "CWT_COLS_VAL=!CWT_COLS_VAL!,!ARG_VAL!"
        ) else (
            REM Space-separated, add with space
            set "CWT_COLS_VAL=!CWT_COLS_VAL! !ARG_VAL!"
        )
    )
    shift
    goto :collect_cwt_cols
    :apply_cwt_cols
    REM Convert comma-separated to space-separated for Python argparse
    set "CWT_COLS_CONVERTED=!CWT_COLS_VAL!"
    set "CWT_COLS_CONVERTED=!CWT_COLS_CONVERTED:,= !"
    set "ARGS=!ARGS! --cwt_cols !CWT_COLS_CONVERTED!"
    goto :parse_args
)
if "%~1"=="--no-plot-block" (
    set "ARGS=!ARGS! --no_plot_block"
    shift
    goto :parse_args
)
if "%~1"=="--no-plot-raw" (
    set "ARGS=!ARGS! --no_plot_raw"
    shift
    goto :parse_args
)
if "%~1"=="--no-plot-ft" (
    set "ARGS=!ARGS! --no_plot_ft"
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
if "%~1"=="--vest-fields" (
    set "ARGS=!ARGS! --vest_fields %~2"
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

:run_analysis
if "!QUERY!"=="" (
    echo ERROR: No query specified.
    echo.
    goto :show_help
)

echo ============================================================================
echo Running IFI Analysis
echo ============================================================================
echo Query: !QUERY!
echo Scheduler: !SCHEDULER!
echo Options: !CWT! !PLOT! !SAVE_DATA! !ARGS!
echo ============================================================================
echo.

python -m ifi.analysis.main_analysis !QUERY! --stft !CWT! --density !PLOT! !SAVE_DATA! --scheduler !SCHEDULER! !ARGS!

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo Analysis completed successfully!
    echo ============================================================================
) else (
    echo.
    echo ============================================================================
    echo Analysis failed with error code: %ERRORLEVEL%
    echo ============================================================================
    exit /b %ERRORLEVEL%
)

goto :end

:show_help
echo IFI Analysis Batch Script
echo.
echo Usage:
echo   run_analysis.bat SHOT_NUMBER [OPTIONS]
echo   run_analysis.bat --query SHOT_NUMBER [OPTIONS]
echo.
echo Query formats:
echo   Single shot: 45687 or --query 45687
echo   Range (colon): "45687:45689" or --query "45687:45689" (processes 45687, 45688, 45689)
echo   List (comma): "45687,45689" or --query "45687,45689" (processes 45687, 45689)
echo.
echo Examples:
echo   run_analysis.bat 45821
echo   run_analysis.bat --query 45821
echo   run_analysis.bat --query "45687:45689"
echo   run_analysis.bat --query "45687,45689"
echo   run_analysis.bat 45687:45689
echo   run_analysis.bat 45687,45689
echo   run_analysis.bat 45821 --cwt
echo   run_analysis.bat 45821 --cwt --plot
echo   run_analysis.bat 45821 --cwt --scheduler processes
echo   run_analysis.bat 45821:45825 --cwt
echo   run_analysis.bat 45821 --cwt --baseline ip
echo   run_analysis.bat 45821 --cwt --color-density-by-amplitude
echo   run_analysis.bat 45821 --cwt --color-density-by-amplitude --amplitude-colormap viridis
echo   run_analysis.bat 45821 --cwt --vest-fields 109 110
echo   run_analysis.bat 45821 --cwt --no-plot-raw --no-plot-ft
echo   run_analysis.bat 45821 --cwt --freq 94
echo   run_analysis.bat 45821 --cwt --freq "94,280"
echo   run_analysis.bat 45821 --cwt --freq "94 280"
echo   run_analysis.bat 45821 --cwt --stft-cols "0,1,2"
echo   run_analysis.bat 45821 --cwt --stft-cols 1
echo   run_analysis.bat 45821 --cwt --stft-cols "0 1 2"
echo.
echo Options:
echo   --cwt                        Enable CWT analysis
echo   --plot                       Enable interactive plotting
echo   --no-plot                    Disable plotting (default)
echo   --scheduler TYPE             Dask scheduler: threads, processes, single-threaded
echo   --processes                  Use processes scheduler (alias)
echo   --single                     Use single-threaded scheduler (alias)
echo   --no-save                    Don't save results to HDF5
echo   --save-plots                 Save plots to files
echo   --force-remote               Force fetching from remote NAS
echo   --baseline TYPE              Baseline correction: ip or trig
echo   --data-folders FOLDERS       Comma-separated list of data folders
echo   --add-path                   Add data folders to default paths
echo   --results-dir DIR             Directory for results (default: ifi/results)
echo   --no-offset-removal          Disable offset removal
echo   --offset-window SIZE         Window size for offset removal (default: 2001)
echo   --stft-cols INDICES          Column indices for STFT
echo                                Default: comma-separated (e.g., "0,1,2")
echo                                Single value: --stft-cols 1
echo                                Space-separated: --stft-cols "0 1 2"
echo   --cwt-cols INDICES           Column indices for CWT
echo                                Default: comma-separated (e.g., "0,1")
echo                                Single value: --cwt-cols 1
echo                                Space-separated: --cwt-cols "0 1"
echo   --no-plot-block              Non-blocking plot mode
echo   --no-plot-raw                Don't plot raw data
echo   --no-plot-ft                 Don't plot time-frequency transforms
echo   --downsample FACTOR          Downsample factor for plotting (default: 10)
echo   --trigger-time SECONDS       Trigger time in seconds (default: 0.290)
echo   --vest-fields FIELDS         Space-separated VEST DB field IDs
echo   --color-density-by-amplitude Color-code density plots by amplitude
echo   --amplitude-colormap MAP     Colormap for amplitude (default: coolwarm)
echo   --amplitude-impedance OHMS   System impedance in ohms (default: 50.0)
echo   --freq FREQ                  Filter to specific frequency(ies): 94, 280, 94.0, 280.0
echo                                Default: comma-separated (e.g., "94,280")
echo                                Single value: --freq 94
echo                                Space-separated: --freq "94 280"
echo   --help                       Show this help message
echo.
goto :end

:end
endlocal


