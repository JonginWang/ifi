@echo off
REM ============================================================================
REM Batch Wrapper for Smart Analysis Runner
REM ============================================================================
REM This script wraps the Python smart analysis runner for convenient batch usage.
REM
REM Usage:
REM   run_analysis_smart.bat 45000:45010 [additional args...]
REM   run_analysis_smart.bat 45000:45010 --freq 280 --density --stft --save_data
REM
REM Example:
REM   run_analysis_smart.bat 45000:45010 --freq 280 --density --stft --stft_cols 0 1 --save_plots --save_data --color_density_by_amplitude --vest_fields 102 109 101 144 214 171
REM ============================================================================

setlocal enabledelayedexpansion

REM Check if shot range is provided
if "%~1"=="" (
    echo ERROR: Shot range is required.
    echo Usage: run_analysis_smart.bat SHOT_RANGE [additional args...]
    echo Example: run_analysis_smart.bat 45000:45010 --freq 280 --density --stft --save_data
    exit /b 1
)

REM Set project root directory
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Get shot range (first argument)
set "SHOT_RANGE=%~1"
shift

REM Collect remaining arguments
set "ADDITIONAL_ARGS="
:collect_args
if "%~1"=="" goto :run_script
set "ADDITIONAL_ARGS=!ADDITIONAL_ARGS! %~1"
shift
goto :collect_args

:run_script
echo ============================================================================
echo Smart Analysis Runner
echo ============================================================================
echo Shot Range: %SHOT_RANGE%
echo Additional Args:%ADDITIONAL_ARGS%
echo Working Directory: %PROJECT_ROOT%
echo ============================================================================
echo.

python "%~dp0run_analysis_smart.py" %SHOT_RANGE%%ADDITIONAL_ARGS%

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

