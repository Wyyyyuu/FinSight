param([switch]$SkipBuild, [int]$ApiPort = 8002)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$nodePath = (Get-Command node.exe -ErrorAction Stop).Source
$runtime = Join-Path $projectRoot 'logs\local-deployment'
$configDir = Join-Path $projectRoot 'data\local-runtime'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Missing .venv; install requirements.txt first.' }
if (-not (Test-Path -LiteralPath (Join-Path $configDir 'user_config.json'))) {
    throw 'Missing local model configuration: data/local-runtime/user_config.json'
}
foreach ($port in @($ApiPort, 5173)) {
    $listener = [System.Net.Sockets.TcpClient]::new()
    try { $listener.Connect('127.0.0.1', $port); throw "Port $port is already in use; stop the existing service first." }
    catch [System.Net.Sockets.SocketException] { }
    finally { $listener.Dispose() }
}
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$selectionPath = Join-Path $projectRoot 'experiments\docker_embedding\selection.json'
if (-not (Test-Path -LiteralPath $selectionPath)) { throw 'Complete the Docker embedding dev evaluation before starting this configuration.' }
$selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
$lexicalModes = @{ m3_dense = 'dense_only'; m3_sparse_rrf = 'legacy'; m3_bm25_rrf = 'bm25' }
if (-not $lexicalModes.ContainsKey($selection.winner)) { throw 'Unsupported embedding evaluation selection.' }
try { $embeddingHealth = Invoke-RestMethod 'http://127.0.0.1:8010/health' -TimeoutSec 5 }
catch { throw 'Start the embedding container first: docker compose -f docker-compose.embedding.yml up -d; wait for its health check.' }
if (-not $embeddingHealth.ready -or $embeddingHealth.model -ne 'bge-m3' -or $embeddingHealth.dim -ne 1024 -or $embeddingHealth.revision -ne $selection.seal.model.revision) {
    throw 'Embedding service does not match the evaluated model profile.'
}
$env:VITE_API_BASE_URL = "http://127.0.0.1:$ApiPort"
$env:VITE_ANNUAL_REPORT_API_BASE_URL = "http://127.0.0.1:$ApiPort"
if (-not $SkipBuild) {
    & npm.cmd run build --prefix (Join-Path $projectRoot 'frontend')
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
}
$env:FINSIGHT_CONFIG_DIR = $configDir
$modelConfig = Get-Content -LiteralPath (Join-Path $configDir 'user_config.json') -Raw | ConvertFrom-Json
$env:ANNUAL_REPORT_LLM_BASE_URL = $modelConfig.llm_api_base
$env:ANNUAL_REPORT_LLM_API_KEY = $modelConfig.llm_api_key
$env:ANNUAL_REPORT_LLM_MODEL = $modelConfig.llm_model
$env:LANGGRAPH_EXECUTE_LIVE_TOOLS = 'true'
$env:ANNUAL_REPORT_EMBEDDING_PROVIDER = 'fastembed'
$env:ANNUAL_REPORT_MODEL_CACHE = Join-Path $projectRoot '.annual-models'
$env:MONITOR_SCAN_ENABLED = 'false'
$env:PRICE_ALERT_SCHEDULER_ENABLED = 'false'
$env:NEWS_ALERT_SCHEDULER_ENABLED = 'false'
$env:RISK_ALERT_SCHEDULER_ENABLED = 'false'
$env:RAG_OBSERVABILITY_RETENTION_ENABLED = 'false'
# Use the prepared local ONNX cross-encoder; no model download is needed.
$env:RAG_EMBEDDING = 'bge-m3-http'
$env:RAG_EMBEDDING_URL = 'http://127.0.0.1:8010'
$env:RAG_REQUIRE_SEMANTIC_EMBEDDING = 'true'
$env:RAG_V2_BACKEND = 'memory'
$env:RAG_V2_VECTOR_DIM = '1024'
$env:RAG_V2_RRF_K = '60'
$env:RAG_LEXICAL_MODE = $lexicalModes[$selection.winner]
$env:RAG_ENABLE_RERANKER = 'true'
$env:RAG_RERANKER = 'bge-reranker-onnx'
$env:RAG_RERANKER_BATCH_SIZE = '4'
$env:RAG_RERANKER_CACHE_DIR = Join-Path $projectRoot 'experiments\lightrag\models'
$env:RAG_V2_TOP_K = '24'
$env:RAG_V2_RERANK_TOP_N = '8'
$env:PYTHONUTF8 = '1'
$backend = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'uvicorn', 'backend.api.main:app', '--host', '127.0.0.1', '--port', "$ApiPort") -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'backend.stdout.log') -RedirectStandardError (Join-Path $runtime 'backend.stderr.log') -PassThru
$frontend = Start-Process -FilePath $nodePath -ArgumentList @('node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', '5173', '--strictPort') -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'frontend.stdout.log') -RedirectStandardError (Join-Path $runtime 'frontend.stderr.log') -PassThru
@{ backendPid = $backend.Id; frontendPid = $frontend.Id; backendStarted = $backend.StartTime.ToUniversalTime().Ticks.ToString(); frontendStarted = $frontend.StartTime.ToUniversalTime().Ticks.ToString(); started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime 'processes.json') -Encoding utf8
$ready = $false
for ($i = 0; $i -lt 45; $i++) {
    try {
        $null = Invoke-WebRequest "http://127.0.0.1:$ApiPort/health" -UseBasicParsing -TimeoutSec 2
        $null = Invoke-WebRequest 'http://127.0.0.1:5173/' -UseBasicParsing -TimeoutSec 2
        $ready = $true
        break
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw "Services did not become ready. See $runtime" }
@{ embedding = $env:RAG_EMBEDDING; profile = $embeddingHealth.revision; backend = $env:RAG_V2_BACKEND; lexicalMode = $env:RAG_LEXICAL_MODE; reranker = $env:RAG_RERANKER; candidates = 24; final = 8; rrfK = 60; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime 'rag-runtime.json') -Encoding utf8
Write-Host 'FinSight: http://127.0.0.1:5173'
Write-Host 'Annual reports: http://127.0.0.1:5173/annual-reports'
Write-Host "API docs: http://127.0.0.1:$ApiPort/docs"
Write-Host "Logs: $runtime"
