@echo off
REM Batch runner for multiple shot ranges using run_analysis.bat
REM Edit the SHOT_LIST line to include your ranges (space-separated)
set "SHOT_LIST="46687:46691" "46692:46695" "46696:46699" "46700:46703" "46595:46599" "46600:46603" "46604:46607" "46608:46611" "46612:46615""

REM Common options for each run (adjust as needed)
set "COMMON_OPTS=--freq 280 --density --stft --stft_cols 0 1 --save_plots --save_data --color_density_by_amplitude --vest_fields 102 109 101 144 214 171"

REM Locate script directory and project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

pushd "%PROJECT_ROOT%"
for %%R in (%SHOT_LIST%) do (
    echo ============================================================
    echo Running analysis for shot(s): %%R
    echo ============================================================
    call "%SCRIPT_DIR%run_analysis.bat" "%%~R" %COMMON_OPTS%
    
    if errorlevel 1 (
        echo [ERROR] run failed for %%R, stopping batch.
        popd
        exit /b 1
    )
)
popd

echo All batch runs completed successfully.
