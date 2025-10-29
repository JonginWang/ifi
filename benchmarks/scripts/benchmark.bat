@echo off
REM ====================================================================
REM IFI Performance Benchmark - System Performance Testing
REM ====================================================================

echo IFI Performance Benchmark Suite
echo ====================================================================
echo Testing system performance with various workloads
echo.

REM Create log directory
if not exist "logs\benchmarks" mkdir "logs\benchmarks"

REM Get timestamp and system info
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%"
set "timestamp=%YY%%MM%%DD%_%HH%%Min%"

echo Benchmark run: %timestamp%
echo.

REM System information
echo System Information:
echo ====================================================================
wmic cpu get name /value | find "Name="
wmic computersystem get TotalPhysicalMemory /value | find "TotalPhysicalMemory="
wmic os get TotalVisibleMemorySize /value | find "TotalVisibleMemorySize="
echo.

REM ====================================================================
REM Benchmark Menu
REM ====================================================================
echo Please select benchmark type:
echo.
echo 1. Quick Benchmark (2-3 minutes)
echo 2. Numba Performance Test
echo 3. Spectrum Analysis Benchmark
echo 4. Memory Usage Test
echo 5. File I/O Performance
echo 6. Full System Stress Test (15+ minutes)
echo 7. Custom Benchmark
echo.
echo 0. Exit
echo.

set /p choice="Enter your choice (0-7): "

if "%choice%"=="0" goto :exit
if "%choice%"=="1" goto :quick_benchmark
if "%choice%"=="2" goto :numba_benchmark
if "%choice%"=="3" goto :spectrum_benchmark
if "%choice%"=="4" goto :memory_benchmark
if "%choice%"=="5" goto :file_io_benchmark
if "%choice%"=="6" goto :stress_test
if "%choice%"=="7" goto :custom_benchmark

echo Invalid choice.
pause
goto :exit

REM ====================================================================
REM Benchmark Suites
REM ====================================================================

:quick_benchmark
echo Quick Benchmark
echo ====================================================================
echo Running essential performance tests...
echo.

call :run_benchmark "Numba JIT Performance" "python -c \"from ifi.test.test_performance import quick_numba_test; quick_numba_test()\"" "quick_numba"
call :run_benchmark "Basic Phase Calculation" "python -c \"from ifi.analysis.phi2ne import PhaseConverter; import numpy as np; pc = PhaseConverter(); t = np.random.randn(10000); print(f'Processed {len(pc.calc_phase_iq_asin2(t, t))} samples')\"" "quick_phase"
call :run_benchmark "FFT Performance" "python -c \"from ifi.analysis.spectrum import SpectrumAnalysis; import numpy as np; sa = SpectrumAnalysis(); signal = np.sin(2*np.pi*10e6*np.arange(100000)/250e6); print(f'Center freq: {sa.find_center_frequency_fft(signal, 250e6)/1e6:.1f} MHz')\"" "quick_fft"

goto :benchmark_complete

:numba_benchmark
echo Numba Performance Benchmark
echo ====================================================================
call :run_benchmark "Numba Comprehensive Test" "python -m ifi.test.test_performance" "numba_comprehensive"
goto :benchmark_complete

:spectrum_benchmark
echo Spectrum Analysis Benchmark
echo ====================================================================
call :run_benchmark "Spectrum Comprehensive Test" "python -m ifi.test.test_spectrum_comprehensive" "spectrum_comprehensive"
goto :benchmark_complete

:memory_benchmark
echo Memory Usage Benchmark
echo ====================================================================
echo Testing memory usage with different data sizes...

REM Memory test with Python
call :run_benchmark "Memory Test - Small (1MB)" "python -c \"import numpy as np; data = np.random.randn(125000); print(f'Allocated {data.nbytes/1e6:.1f} MB')\"" "memory_small"
call :run_benchmark "Memory Test - Medium (100MB)" "python -c \"import numpy as np; data = np.random.randn(12500000); print(f'Allocated {data.nbytes/1e6:.1f} MB')\"" "memory_medium"
call :run_benchmark "Memory Test - Large (1GB)" "python -c \"import numpy as np; data = np.random.randn(125000000); print(f'Allocated {data.nbytes/1e9:.1f} GB')\"" "memory_large"

goto :benchmark_complete

:file_io_benchmark
echo File I/O Performance Benchmark
echo ====================================================================
echo Testing file I/O performance...

REM Create test directory
if not exist "temp_benchmark" mkdir "temp_benchmark"

call :run_benchmark "File Write Test" "python -c \"import numpy as np; import pandas as pd; df = pd.DataFrame({'TIME': np.arange(1000000)/250e6, 'CH0': np.random.randn(1000000)}); df.to_csv('temp_benchmark/test_write.csv', index=False); print('Wrote 1M samples')\"" "file_write"
call :run_benchmark "File Read Test" "python -c \"import pandas as pd; df = pd.read_csv('temp_benchmark/test_write.csv'); print(f'Read {len(df):,} samples')\"" "file_read"

REM Cleanup
if exist "temp_benchmark" rmdir /s /q "temp_benchmark"

goto :benchmark_complete

:stress_test
echo Full System Stress Test
echo ====================================================================
echo Running comprehensive system stress test...
echo This will take 15+ minutes. Continue? (Y/N)
set /p confirm="Continue? "
if /i not "%confirm%"=="y" goto :benchmark_complete

echo.
echo Starting stress test...

call :run_benchmark "Integration Full Test" "python -m ifi.test.test_integration_full" "stress_integration"
call :run_benchmark "Spectrum Comprehensive" "python -m ifi.test.test_spectrum_comprehensive" "stress_spectrum"
call :run_benchmark "Plots Comprehensive" "python -m ifi.test.test_plots_comprehensive" "stress_plots"
call :run_benchmark "Phi2ne Comprehensive" "python -m ifi.test.test_phi2ne_comprehensive" "stress_phi2ne"

echo.
echo CPU stress test with real data analysis...
call :run_benchmark "Real Data Analysis" "python -m ifi.analysis.main_analysis 45821 --density --vest_fields 109 101" "stress_real_data"

goto :benchmark_complete

:custom_benchmark
echo Custom Benchmark
echo ====================================================================
echo Enter Python command to benchmark:
set /p custom_cmd="Command: "
if "%custom_cmd%"=="" goto :benchmark_complete

echo Enter benchmark name:
set /p bench_name="Name: "
if "%bench_name%"=="" set bench_name=custom

call :run_benchmark "Custom: %bench_name%" "%custom_cmd%" "custom_%bench_name%"
goto :benchmark_complete

REM ====================================================================
REM Helper Functions
REM ====================================================================

:run_benchmark
set bench_name=%~1
set bench_cmd=%~2
set log_suffix=%~3

echo.
echo Running: %bench_name%
echo --------------------------------------------------------------------
echo Command: %bench_cmd%
echo.

REM Record start time
set start_time=%time%
echo Start time: %start_time%

REM Run benchmark and capture output
%bench_cmd% > "logs\benchmarks\%timestamp%_%log_suffix%.log" 2>&1

REM Record end time and calculate duration
set end_time=%time%
echo End time: %end_time%

if %errorlevel% equ 0 (
    echo %bench_name% COMPLETED
) else (
    echo %bench_name% FAILED (Exit code: %errorlevel%)
    echo    Check log: logs\benchmarks\%timestamp%_%log_suffix%.log
)

echo Duration: %start_time% to %end_time%
echo.
goto :eof

:benchmark_complete
echo.
echo ====================================================================
echo Benchmark completed!
echo ====================================================================
echo.
echo Benchmark logs saved in: logs\benchmarks\
echo Performance data available for analysis
echo.

REM System resource summary
echo System Resource Summary:
echo ====================================================================
wmic process where name="python.exe" get ProcessId,PageFileUsage,WorkingSetSize /format:table
echo.

echo Would you like to:
echo 1. Run another benchmark
echo 2. View benchmark logs
echo 3. Generate performance report
echo 4. Exit
echo.

set /p next_choice="Enter choice (1-4): "

if "%next_choice%"=="1" goto :start
if "%next_choice%"=="2" start explorer "logs\benchmarks"
if "%next_choice%"=="3" goto :generate_report
if "%next_choice%"=="4" goto :exit

:generate_report
echo Generating Performance Report...
echo ====================================================================
echo Performance report functionality can be added here
echo Current logs location: logs\benchmarks\%timestamp%_*.log
echo.
pause
goto :exit

:exit
echo.
echo Benchmark testing completed!
echo Use the log files to analyze system performance
pause
exit /b 0