$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$dashboardPort = if ($env:PHASE8_DASHBOARD_PORT) {
    $env:PHASE8_DASHBOARD_PORT
} else {
    '8501'
}
Push-Location $projectRoot
try {
    python -m streamlit run phase8/app/dashboard.py `
        --server.port $dashboardPort `
        --browser.gatherUsageStats false
} finally {
    Pop-Location
}
