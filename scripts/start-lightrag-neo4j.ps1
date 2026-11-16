param([switch]$SkipVerify)
$ErrorActionPreference = 'Stop'
$neoProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $neoProjectRoot
try {
    $neoPython = Join-Path $neoProjectRoot 'experiments\lightrag\.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $neoPython)) {
        throw 'Missing isolated LightRAG venv. See experiments/lightrag/NEO4J.md.'
    }
    & $neoPython experiments/lightrag/neo4j_experiment.py init
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize Neo4j configuration' }
    & docker compose --env-file experiments/lightrag/.env.neo4j.local -f docker-compose.lightrag.yml up -d --wait --wait-timeout 240
    if ($LASTEXITCODE -ne 0) { throw 'Neo4j startup failed; check Docker Desktop and container health' }
    & $neoPython experiments/lightrag/neo4j_experiment.py migrate
    if ($LASTEXITCODE -ne 0) { throw 'Graph migration incomplete; original index is preserved' }
    if (-not $SkipVerify) {
        & $neoPython experiments/lightrag/neo4j_experiment.py verify
        if ($LASTEXITCODE -ne 0) { throw 'Neo4j verification failed; see verification.json' }
    }
    Write-Host 'Neo4j Browser: http://127.0.0.1:7474'
    Write-Host 'Credentials: experiments/lightrag/.env.neo4j.local (local only)'
} finally {
    Pop-Location
}
