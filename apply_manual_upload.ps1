param(
    [Parameter(Mandatory=$true)]
    [string]$RepositoryPath
)

$ErrorActionPreference = "Stop"
$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = (Resolve-Path $RepositoryPath).Path

Write-Host "Source bundle: $Source"
Write-Host "Target repo:    $Target"

$remove = @(
    "bootstrap",
    ".bootstrap",
    ".runtime_payload",
    "src\gp3mlpy\__pycache__",
    "tests\__pycache__",
    ".pytest_cache"
)
foreach ($rel in $remove) {
    $p = Join-Path $Target $rel
    if (Test-Path $p) { Remove-Item $p -Recurse -Force }
}

Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
    if ($_.Name -ne "apply_manual_upload.ps1") {
        Copy-Item -LiteralPath $_.FullName -Destination $Target -Recurse -Force
    }
}

Write-Host "Bundle copied. Review 'git status' before committing."
