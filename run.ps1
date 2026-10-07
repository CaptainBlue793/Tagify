param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & conda run --no-capture-output -n starGPU python -m streamlit run Tagify.py --server.port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
