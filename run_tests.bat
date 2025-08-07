@echo off
REM ====================================================================
REM IFI Test Suite - Automated Batch Runner
REM ====================================================================

echo IFI Comprehensive Test Suite
echo ====================================================================
echo.

set TEST_DIR=ifi\results\test_outputs
set LOG_DIR=logs\batch_tests

REM Create output directories
if not exist "%TEST_DIR%" mkdir "%TEST_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Get current timestamp for log files
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YY%%MM%%DD%_%HH%%Min%%Sec%"

echo Test run timestamp: %timestamp%
echo.

REM ====================================================================
REM Test Menu
REM ====================================================================
echo Please select test suite to run:
echo.
echo 1. Quick Tests (Performance + Basic)
echo 2. Spectrum Analysis Tests (STFT, CWT)
echo 3. Plotting Tests (All plots)
echo 4. Integration Tests (Full pipeline)
echo 5. Main Analysis Tests (Real data)
echo 6. Database Tests
echo 7. ALL TESTS (Complete suite)
echo 8. Custom Test (Enter module name)
echo.
echo 0. Exit
echo.

set /p choice="Enter your choice (0-8): "

if "%choice%"=="0" goto :exit
if "%choice%"=="1" goto :quick_tests
if "%choice%"=="2" goto :spectrum_tests
if "%choice%"=="3" goto :plot_tests
if "%choice%"=="4" goto :integration_tests
if "%choice%"=="5" goto :main_analysis_tests
if "%choice%"=="6" goto :database_tests
if "%choice%"=="7" goto :all_tests
if "%choice%"=="8" goto :custom_test

echo Invalid choice. Please try again.
pause
goto :exit

REM ====================================================================
REM Test Suites
REM ====================================================================

:quick_tests
echo Running Quick Tests...
echo ====================================================================
call :run_single_test "Performance Tests" "ifi.test.test_performance"
call :run_single_test "Phi2ne Comprehensive" "ifi.test.test_phi2ne_comprehensive"
call :run_single_test "Interferometry Params" "ifi.test.test_interferometry_params"
goto :test_complete

:spectrum_tests
echo Running Spectrum Analysis Tests...
echo ====================================================================
call :run_single_test "Spectrum Simple" "ifi.test.test_spectrum_simple"
call :run_single_test "Spectrum Comprehensive" "ifi.test.test_spectrum_comprehensive"
goto :test_complete

:plot_tests
echo Running Plotting Tests...
echo ====================================================================
call :run_single_test "Plots Comprehensive" "ifi.test.test_plots_comprehensive"
goto :test_complete

:integration_tests
echo Running Integration Tests...
echo ====================================================================
call :run_single_test "Integration Full" "ifi.test.test_integration_full"
call :run_single_test "Main Analysis" "ifi.test.test_main_analysis"
goto :test_complete

:main_analysis_tests
echo Running Main Analysis Tests with Real Data...
echo ====================================================================
echo Enter shot number (e.g., 45821):
set /p shot_num="Shot number: "
if "%shot_num%"=="" set shot_num=45821

echo.
echo Running main analysis for shot %shot_num%...
echo Command: python -m ifi.analysis.main_analysis %shot_num% --density --vest_fields 109 101
echo.
python -m ifi.analysis.main_analysis %shot_num% --density --vest_fields 109 101
echo.
echo Main analysis completed.
goto :test_complete

:database_tests
echo Running Database Tests...
echo ====================================================================
call :run_single_test "Database Tests" "ifi.test.test_db"
goto :test_complete

:all_tests
echo Running ALL Tests...
echo ====================================================================
call :run_single_test "Performance Tests" "ifi.test.test_performance"
call :run_single_test "Phi2ne Comprehensive" "ifi.test.test_phi2ne_comprehensive"
call :run_single_test "Interferometry Params" "ifi.test.test_interferometry_params"
call :run_single_test "Spectrum Simple" "ifi.test.test_spectrum_simple"
call :run_single_test "Spectrum Comprehensive" "ifi.test.test_spectrum_comprehensive"
call :run_single_test "Plots Comprehensive" "ifi.test.test_plots_comprehensive"
call :run_single_test "Integration Full" "ifi.test.test_integration_full"
call :run_single_test "Main Analysis" "ifi.test.test_main_analysis"
call :run_single_test "Database Tests" "ifi.test.test_db"
goto :test_complete

:custom_test
echo Custom Test...
echo ====================================================================
echo Enter test module name (e.g., ifi.test.test_spectrum_simple):
set /p custom_module="Module: "
if "%custom_module%"=="" goto :test_complete

call :run_single_test "Custom Test" "%custom_module%"
goto :test_complete

REM ====================================================================
REM Helper Functions
REM ====================================================================

:run_single_test
echo.
echo Running %~1...
echo --------------------------------------------------------------------
set start_time=%time%
echo Start time: %start_time%
echo Command: python -m %~2
echo.

python -m %~2 > "%LOG_DIR%\%timestamp%_%~1.log" 2>&1

if %errorlevel% equ 0 (
    echo %~1 PASSED
) else (
    echo %~1 FAILED (Exit code: %errorlevel%)
    echo    Check log: %LOG_DIR%\%timestamp%_%~1.log
)

set end_time=%time%
echo End time: %end_time%
echo.
goto :eof

:test_complete
echo.
echo ====================================================================
echo Test suite completed!
echo ====================================================================
echo.
echo Log files saved in: %LOG_DIR%\
echo Test outputs saved in: %TEST_DIR%\
echo.
echo Would you like to:
echo 1. Run another test suite
echo 2. View log directory
echo 3. View results directory
echo 4. Exit
echo.

set /p next_choice="Enter choice (1-4): "

if "%next_choice%"=="1" goto :start
if "%next_choice%"=="2" start explorer "%LOG_DIR%"
if "%next_choice%"=="3" start explorer "%TEST_DIR%"
if "%next_choice%"=="4" goto :exit

:exit
echo.
echo Thanks for using IFI Test Suite!
pause
exit /b 0