# ==============================================================================
# PowerShell Batch Runner for Spectrum Analysis
# ==============================================================================

# 1. 샷 리스트 정의 (콜론 사용해도 문제 없음)
$shotList = @(
    "46692:46695", "46696:46699", "46700:46703",
    "46595:46599", "46600:46603", "46604:46607", "46608:46611", "46612:46615"
)

# 2. 공통 인수 설정 (명령어 뒷부분)
$commonArgs = "--freq 280 --density --stft --stft_cols 1 2 --save_plots --save_data --color_density_by_amplitude --vest_fields 102 109 101 144 214 171"

# 3. 경로 설정
$scriptDir = $PSScriptRoot
# ".."을 사용하여 상위 폴더(프로젝트 루트) 경로를 절대 경로로 변환
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

# 4. 실행 루프
foreach ($range in $shotList) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Running analysis for shot(s): $range" -ForegroundColor Green
    Write-Host "Working Directory: $projectRoot" -ForegroundColor Gray
    Write-Host "============================================================"

    # 실행할 전체 인수 문자열 구성
    # 형식: -m ifi.analysis.main_analysis [RANGE] [COMMON_ARGS]
    $procArgs = "-m ifi.analysis.main_analysis $range $commonArgs"

    # 5. 프로세스 시작 (핵심 부분)
    # -FilePath: 실행할 프로그램 (python)
    # -ArgumentList: 전달할 인수들
    # -WorkingDirectory: 실행 위치 ($projectRoot)
    # -Wait: 작업이 끝날 때까지 대기
    # -NoNewWindow: 새 창을 띄우지 않고 현재 창에서 실행
    # -PassThru: 프로세스 정보를 리턴받아 에러 코드 확인
    
    $process = Start-Process -FilePath "python" `
                             -ArgumentList $procArgs `
                             -WorkingDirectory $projectRoot `
                             -Wait `
                             -NoNewWindow `
                             -PassThru

    # 6. 에러 체크
    if ($process.ExitCode -ne 0) {
        Write-Host "[ERROR] Analysis failed for range: $range (Exit Code: $($process.ExitCode))" -ForegroundColor Red
        # 에러 발생 시 멈추고 싶다면 아래 줄 주석 해제
        # exit 1 
    }
}

Write-Host "`nAll batch runs completed." -ForegroundColor Cyan