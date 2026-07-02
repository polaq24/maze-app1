<#
.SYNOPSIS
    Genera report snapshot dello stato corrente del terminale Maze Capital.
.DESCRIPTION
    Legge i dati da macro_history.db e produce un report .md nella cartella corrente.
.PARAMETER Force
    Forza sovrascrittura del report esistente.
.PARAMETER Now
    Genera il report immediatamente (obbligatorio insieme a -Force).
.EXAMPLE
    .\snapshot.ps1 -Force -Now
#>
param(
    [switch]$Force,
    [switch]$Now
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyScript = Join-Path $scriptDir "snapshot.py"

if (-not (Test-Path $pyScript)) {
    Write-Host "❌ snapshot.py non trovato in: $scriptDir" -ForegroundColor Red
    exit 1
}

if (-not $Force -or -not $Now) {
    Write-Host "Usa: .\snapshot.ps1 -Force -Now" -ForegroundColor Yellow
    exit 1
}

Write-Host "⬡ Maze Capital — Generazione snapshot..." -ForegroundColor Cyan

python $pyScript --force --now

if ($LASTEXITCODE -eq 0) {
    $latest = Get-ChildItem -Path $scriptDir -Filter "snapshot_*.md" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        Write-Host "✅ Report: $($latest.FullName)" -ForegroundColor Green
        Write-Host "   Apri con: notepad $($latest.Name)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "❌ Errore durante la generazione" -ForegroundColor Red
}
