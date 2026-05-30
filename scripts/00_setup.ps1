# Set up the local environment: install CPU-only dependencies into the .venv
# and verify Kaggle credentials. Run from anywhere:  .\scripts\00_setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pip = Join-Path $root ".venv\Scripts\pip.exe"
$kaggle = Join-Path $root ".venv\Scripts\kaggle.exe"

Write-Host "Installing local dependencies into .venv ..." -ForegroundColor Cyan
& $pip install -r (Join-Path $root "requirements-local.txt")

Write-Host "`nChecking Kaggle credentials ..." -ForegroundColor Cyan
$cred = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (Test-Path $cred) {
    Write-Host "Found kaggle.json at $cred"
    & $kaggle kernels list -m --page-size 1 | Out-Null
    if ($?) { Write-Host "Kaggle auth OK." -ForegroundColor Green }
    else { Write-Host "Kaggle auth FAILED - verify the token in kaggle.json." -ForegroundColor Yellow }
} else {
    Write-Host "kaggle.json NOT found at $cred" -ForegroundColor Yellow
    Write-Host "  1. Go to kaggle.com -> Settings -> API -> 'Create New Token'."
    Write-Host "  2. Move the downloaded kaggle.json to: $cred"
    Write-Host "  3. Re-run this script."
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  python scripts\01_prepare_data.py      # download SIGMORPHON data"
Write-Host "  python scripts\02_make_bundle.py       # upload code+data as a Kaggle dataset"
Write-Host "  python scripts\03_push_and_poll.py     # run the GPU kernels and pull results"
Write-Host "  python build_artifacts.py              # build tables + figures"
