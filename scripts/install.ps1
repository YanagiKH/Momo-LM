$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VenvPath "Scripts\python.exe"
$Momo = Join-Path $VenvPath "Scripts\momo.exe"

python -m venv $VenvPath
& $Python -m pip install --upgrade pip
& $Python -m pip install -e $ProjectRoot
& $Momo init

Write-Host "Momo-LM installed. Start it with:"
Write-Host "  $Momo serve"
