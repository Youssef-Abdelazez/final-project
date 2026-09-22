$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $projectRoot
try {
    python -m pytest phase8/tests -q
} finally {
    Pop-Location
}
