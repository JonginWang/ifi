@echo off
REM ====================================================================
REM IFI Real Data Test - Test with Various Shot Numbers
REM ====================================================================

echo IFI Real Data Test Suite
echo ====================================================================
echo Testing main analysis with different shot numbers and configurations
echo.

REM Create log directory
if not exist "logs\real_data_tests" mkdir "logs\real_data_tests"

REM Get timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%"
set "timestamp=%YY%%MM%%DD%_%HH%%Min%"

echo 📅 Real data test run: %timestamp%
echo.

REM ====================================================================
REM Test Configurations
REM ====================================================================

echo Please choose test configuration:
echo.
echo 1. Single Shot Test (specify shot number)
echo 2. Standard Test Shots (45821, 45822, 45823)
echo 3. Historical Shots (40000, 35000)
echo 4. Custom Multiple Shots
echo 5. Density Analysis Focus
echo 6. VEST Field Comparison
echo.
echo 0. Exit
echo.

set /p choice="Enter your choice (0-6): "

if "%choice%"=="0" goto :exit
if "%choice%"=="1" goto :single_shot
if "%choice%"=="2" goto :standard_shots
if "%choice%"=="3" goto :historical_shots
if "%choice%"=="4" goto :custom_shots
if "%choice%"=="5" goto :density_focus
if "%choice%"=="6" goto :vest_comparison

echo Invalid choice.
pause
goto :exit

REM ====================================================================
REM Test Scenarios
REM ====================================================================

:single_shot
echo Single Shot Test
echo ====================================================================
echo Enter shot number:
set /p shot_num="Shot number: "
if "%shot_num%"=="" goto :exit

echo.
echo Testing shot %shot_num% with density analysis...
call :run_analysis "%shot_num%" "--density --vest_fields 109 101" "single_%shot_num%"

echo.
echo Testing shot %shot_num% without density analysis...
call :run_analysis "%shot_num%" "--vest_fields 109 101" "basic_%shot_num%"

goto :test_complete

:standard_shots
echo Standard Test Shots
echo ====================================================================
call :run_analysis "45821" "--density --vest_fields 109 101" "std_45821"
call :run_analysis "45822" "--density --vest_fields 109 101" "std_45822"
call :run_analysis "45823" "--density --vest_fields 109 101" "std_45823"
goto :test_complete

:historical_shots
echo Historical Shots
echo ====================================================================
call :run_analysis "40000" "--vest_fields 109 101" "hist_40000"
call :run_analysis "35000" "--vest_fields 109 101" "hist_35000"
goto :test_complete

:custom_shots
echo Custom Multiple Shots
echo ====================================================================
echo Enter shot numbers separated by spaces (e.g., 45821 45822 45823):
set /p shot_list="Shot numbers: "
if "%shot_list%"=="" goto :exit

echo.
echo Processing shots: %shot_list%
for %%s in (%shot_list%) do (
    echo Testing shot %%s...
    call :run_analysis "%%s" "--density --vest_fields 109 101" "custom_%%s"
)
goto :test_complete

:density_focus
echo Density Analysis Focus
echo ====================================================================
echo Enter shot number for detailed density analysis:
set /p shot_num="Shot number: "
if "%shot_num%"=="" set shot_num=45821

echo.
echo Running comprehensive density analysis for shot %shot_num%...

REM Test different VEST field combinations
call :run_analysis "%shot_num%" "--density --vest_fields 109" "density_ip_%shot_num%"
call :run_analysis "%shot_num%" "--density --vest_fields 101" "density_halpha_%shot_num%"
call :run_analysis "%shot_num%" "--density --vest_fields 109 101" "density_both_%shot_num%"
call :run_analysis "%shot_num%" "--density" "density_only_%shot_num%"

goto :test_complete

:vest_comparison
echo VEST Field Comparison
echo ====================================================================
echo Enter shot number for VEST field comparison:
set /p shot_num="Shot number: "
if "%shot_num%"=="" set shot_num=45821

echo.
echo Testing different VEST field combinations for shot %shot_num%...

REM Different field combinations
call :run_analysis "%shot_num%" "--vest_fields 109" "vest_ip_%shot_num%"
call :run_analysis "%shot_num%" "--vest_fields 101" "vest_halpha_%shot_num%"
call :run_analysis "%shot_num%" "--vest_fields 109 101" "vest_both_%shot_num%"
call :run_analysis "%shot_num%" "--vest_fields 109 101 102" "vest_multi_%shot_num%"

goto :test_complete

REM ====================================================================
REM Helper Functions
REM ====================================================================

:run_analysis
set shot=%~1
set params=%~2
set log_suffix=%~3

echo.
echo Running analysis: Shot %shot% with parameters %params%
echo --------------------------------------------------------------------
set start_time=%time%
echo Start time: %start_time%
echo Command: python -m ifi.analysis.main_analysis %shot% %params%
echo.

python -m ifi.analysis.main_analysis %shot% %params% > "logs\real_data_tests\%timestamp%_%log_suffix%.log" 2>&1

if %errorlevel% equ 0 (
    echo Analysis PASSED for shot %shot%
) else (
    echo Analysis FAILED for shot %shot% (Exit code: %errorlevel%)
    echo    Check log: logs\real_data_tests\%timestamp%_%log_suffix%.log
)

set end_time=%time%
echo End time: %end_time%
echo.
goto :eof

:test_complete
echo.
echo ====================================================================
echo Real data tests completed!
echo ====================================================================
echo.
echo Log files saved in: logs\real_data_tests\
echo Analysis results saved in: ifi\results\
echo.
echo Would you like to:
echo 1. Run another test configuration
echo 2. View log directory
echo 3. View results directory
echo 4. Exit
echo.

set /p next_choice="Enter choice (1-4): "

if "%next_choice%"=="1" goto :start
if "%next_choice%"=="2" start explorer "logs\real_data_tests"
if "%next_choice%"=="3" start explorer "ifi\results"
if "%next_choice%"=="4" goto :exit

:exit
echo.
echo Real data testing completed!
pause
exit /b 0