$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo

function Run-Step([string]$Name, [scriptblock]$Command) {
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Run-Step "PYTHON" { uv run python --version }
Run-Step "PYTEST" { uv run python -m pytest -q }
Run-Step "RELEASE VALIDATION" { uv run python scripts\validate_release.py }
Run-Step "COMPILE" { uv run python -m compileall -q src\gp3mlpy tests examples scripts }
Run-Step "RUFF" { uv run python -m ruff check src tests examples }
Run-Step "MYPY PUBLIC STUB" { uv run python -m mypy --follow-imports=skip src\gp3mlpy\__init__.pyi }
Run-Step "MKDOCS STRICT" { uv run python -m mkdocs build --strict }

Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
Run-Step "BUILD" { uv run python -m build }
Run-Step "TWINE" { uv run python -m twine check dist\* }

Write-Host ""
Write-Host "=== FROZEN API ===" -ForegroundColor Cyan
uv run python -c "import gp3mlpy as gp; x=gp.gp3ml_api_contracts(); assert len(x.exports)==127; assert int((x.exports['stability']=='stable').sum())==71; assert int((x.exports['stability']=='experimental').sum())==56; assert gp.r_reference_version=='0.3.0'; print('127 / 71 / 56 / R 0.3.0: PASS')"
if ($LASTEXITCODE -ne 0) { throw "Frozen API validation failed." }

Write-Host ""
Write-Host "ALL QUALITY GATES PASSED" -ForegroundColor Green
