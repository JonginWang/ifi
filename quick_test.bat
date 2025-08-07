@echo off
REM ====================================================================
REM IFI Quick Test - Essential Tests Only
REM ====================================================================

echo IFI Quick Test Suite
echo ====================================================================
echo Running essential tests for rapid verification...
echo.

REM Create log directory
if not exist "logs\quick_tests" mkdir "logs\quick_tests"

REM Get timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%"
set "timestamp=%YY%%MM%%DD%_%HH%%Min%"

echo Quick test run: %timestamp%
echo.

REM ====================================================================
REM Quick Test Sequence
REM ====================================================================

echo 1/4 Testing Numba Performance...
python -m ifi.test.test_performance > "logs\quick_tests\%timestamp%_performance.log" 2>&1
if %errorlevel% equ 0 (echo Performance test PASSED) else (echo Performance test FAILED)

echo.
echo 2/4 Testing Interferometry Parameters...
python -m ifi.test.test_interferometry_params > "logs\quick_tests\%timestamp%_params.log" 2>&1
if %errorlevel% equ 0 (echo Parameters test PASSED) else (echo Parameters test FAILED)

echo.
echo 3/4 Testing Basic Spectrum Analysis...
python -m ifi.test.test_spectrum_simple > "logs\quick_tests\%timestamp%_spectrum.log" 2>&1
if %errorlevel% equ 0 (echo Spectrum test PASSED) else (echo Spectrum test FAILED)

echo.
echo 4/4 Testing Phi2ne Module...
python -c "from ifi.analysis.phi2ne import PhaseConverter, get_interferometry_params; print('Phi2ne import successful')" > "logs\quick_tests\%timestamp%_phi2ne.log" 2>&1
if %errorlevel% equ 0 (echo Phi2ne test PASSED) else (echo Phi2ne test FAILED)

echo.
echo ====================================================================
echo Quick tests completed in under 1 minute!
echo Logs saved in: logs\quick_tests\
echo.
echo Run 'run_tests.bat' for comprehensive testing.
echo.
pause