$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $projectRoot
try {
    python -m uvicorn phase8.app.api:app --host 127.0.0.1 --port 8000
} finally {
    Pop-Location
}
