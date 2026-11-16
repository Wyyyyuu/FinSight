param([int]$Port = 8001, [switch]$Semantic, [switch]$Intent)
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
if ($Intent) {
    $intentModel = if ($env:ANNUAL_REPORT_INTENT_MODEL_DIR) {
        $env:ANNUAL_REPORT_INTENT_MODEL_DIR
    } else {
        Join-Path $projectRoot '.annual-models\annual-intent-minirbt-v3'
    }
    $missing = @('model.onnx', 'tokenizer.json', 'metadata.json') | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $intentModel $_) -PathType Leaf)
    }
    if ($missing) {
        throw '本地意图模型未就绪。请按 docs/ANNUAL_REPORTS.md 的步骤训练和导出。'
    }
    $env:ANNUAL_REPORT_INTENT_MODEL_DIR = $intentModel
}
Set-Location -LiteralPath $projectRoot
Write-Host "年报工作台：http://127.0.0.1:$Port/annual-reports"
& $pythonPath -m uvicorn backend.annual_reports.app:app --host 127.0.0.1 --port $Port
