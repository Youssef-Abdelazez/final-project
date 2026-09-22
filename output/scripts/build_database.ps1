$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $projectRoot
try {
    python -m phase8.app.import_data
} finally {
    Pop-Location
}
