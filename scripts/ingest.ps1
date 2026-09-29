# 从项目根目录运行，并优先使用本地虚拟环境
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        & $venvPython -m docpulse ingest @args
    } else {
        python -m docpulse ingest @args
    }
} finally {
    Pop-Location
}