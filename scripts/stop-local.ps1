$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $projectRoot 'logs\local-deployment\processes.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Host 'No local deployment process record.'; return }
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
foreach ($service in @('frontend', 'backend')) {
    $process = Get-Process -Id $state."${service}Pid" -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    if ($process.StartTime.ToUniversalTime().Ticks.ToString() -ne $state."${service}Started") {
        throw "The $service PID belongs to a different process; refusing to stop it."
    }
    & taskkill.exe /PID $process.Id /T /F
    if ($LASTEXITCODE -ne 0) { throw "Could not stop $service." }
}
Write-Host 'Local FinSight services stopped.'
