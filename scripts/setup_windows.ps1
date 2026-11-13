$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Instale Python 3.11 (64 bits) antes de continuar.' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw 'FFmpeg nao encontrado no PATH. Instale com: winget install Gyan.FFmpeg' }
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host 'Instalacao concluida. Inicie com: .\.venv\Scripts\python.exe -m sampleforge'
