@echo off
REM ============================================================================
REM Batch Wrapper for Smart Analysis Runner
REM ============================================================================
REM This script wraps the Python smart analysis runner for convenient batch usage.
REM Supports multiple shot ranges via --shot-list option.
REM
REM Usage:
REM   Single range: run_analysis_smart.bat 45000:45010 [additional args...]
REM   Multiple ranges: run_analysis_smart.bat --shot-list "45000:45010" "45020:45030" [additional args...]
REM   Example list: run_analysis_smart.bat [additional args...] (uses example list)
REM
REM Example:
REM   run_analysis_smart.bat 45000:45010 --freq 280 --density --stft --save_data
REM   run_analysis_smart.bat --shot-list "45000:45010" "45020:45030" --freq 280 --density --stft --save_data
REM ============================================================================

setlocal enabledelayedexpansion

REM Set project root directory
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Parse command line arguments
set "SHOT_LIST="
set "ARGS="
set "USE_SINGLE_RANGE=0"
set "SINGLE_RANGE="

:parse_args
if "%~1"=="" goto :run_script
if "%~1"=="--help" (
    goto :show_help
)
if "%~1"=="--shot-list" (
    REM Collect shot list values
    set "SHOT_LIST="
    set "USE_SINGLE_RANGE=0"
    shift
    :collect_shots
    if "%~1"=="" goto :parse_args
    REM Check if next argument starts with -- (another option), if so stop collecting
    set "ARG_VAL=%~1"
    echo !ARG_VAL! | findstr /R "^--" >nul
    if !ERRORLEVEL! EQU 0 goto :parse_args
    REM Add shot range to list
    if "!SHOT_LIST!"=="" (
        set "SHOT_LIST=--shot-list !ARG_VAL!"
    ) else (
        set "SHOT_LIST=!SHOT_LIST! !ARG_VAL!"
    )
    shift
    goto :collect_shots
)
REM Check if first argument is a shot range (not starting with --)
if "!USE_SINGLE_RANGE!"=="0" (
    echo %~1 | findstr /R "^--" >nul
    if !ERRORLEVEL! NEQ 0 (
        set "SINGLE_RANGE=%~1"
        set "USE_SINGLE_RANGE=1"
        shift
        goto :parse_args
    )
)
REM All other arguments are passthrough
set "ARGS=!ARGS! %~1"
shift
goto :parse_args

:run_script
echo ============================================================================
echo Smart Analysis Runner
echo ============================================================================

REM Build command
set "CMD=python "%~dp0run_analysis_smart.py""

if "!USE_SINGLE_RANGE!"=="1" (
    echo Shot Range: !SINGLE_RANGE!
    set "CMD=!CMD! !SINGLE_RANGE!"
) else if not "!SHOT_LIST!"=="" (
    echo Shot Ranges: !SHOT_LIST!
    set "CMD=!CMD! !SHOT_LIST!"
) else (
    echo Using example shot list (use --shot-list to override)
    echo.
)

if not "!ARGS!"=="" (
    echo Additional Args: !ARGS!
    set "CMD=!CMD!!ARGS!"
)

echo Working Directory: %PROJECT_ROOT%
echo ============================================================================
echo.

!CMD!

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo Smart analysis completed successfully!
    echo ============================================================================
    exit /b 0
) else (
    echo.
    echo ============================================================================
    echo Smart analysis failed with error code: %ERRORLEVEL%
    echo ============================================================================
    exit /b %ERRORLEVEL%
)

goto :end

:show_help
echo Smart Analysis Runner - Batch Wrapper
echo.
echo Usage:
echo   Single range: run_analysis_smart.bat SHOT_RANGE [OPTIONS]
echo   Multiple ranges: run_analysis_smart.bat --shot-list "RANGE1" "RANGE2" [OPTIONS]
echo   Example list: run_analysis_smart.bat [OPTIONS] (uses example list)
echo.
echo Examples:
echo   run_analysis_smart.bat 45000:45010 --freq 280 --density --stft --save_data
echo   run_analysis_smart.bat --shot-list "45000:45010" "45020:45030" --freq 280 --density --stft --save_data
echo   run_analysis_smart.bat --freq 280 --density --stft --save_data
echo.
echo Options:
echo   --shot-list RANGES          Space-separated shot ranges (e.g., "45000:45010" "45020:45030")
echo                               If not provided, uses example list from script
echo   All other arguments are passed to the Python script
echo   --help                      Show this help message
echo.
goto :end

:end
endlocal
