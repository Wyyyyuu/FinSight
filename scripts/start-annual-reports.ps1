param([int]$Port = 8001, [switch]$Semantic)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '请先在仓库根目录运行 python -m venv .venv，并安装 requirements-annual.txt。'
}
if ($Semantic) {
    $env:ANNUAL_REPORT_EMBEDDING_PROVIDER = 'fastembed'
    if (-not $env:ANNUAL_REPORT_MODEL_CACHE) {
        $env:ANNUAL_REPORT_MODEL_CACHE = Join-Path $projectRoot '.annual-models'
    }
}
Set-Location -LiteralPath $projectRoot
Write-Host "年报工作台：http://127.0.0.1:$Port/annual-reports"
& $pythonPath -m uvicorn backend.annual_reports.app:app --host 127.0.0.1 --port $Port
