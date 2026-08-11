param(
    [switch]$SkipInstall,
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$spec = Join-Path $PSScriptRoot "KnowledgeAssistant.spec"
$distDirectory = Join-Path $projectRoot "dist\KnowledgeAssistant"
$releaseDirectory = Join-Path $projectRoot "release"
$version = (Get-Content -Raw (Join-Path $projectRoot "VERSION")).Trim()
$archive = Join-Path $releaseDirectory "KnowledgeAssistant-Windows-x64-v$version.zip"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing virtual environment: $python"
}

Set-Location -LiteralPath $projectRoot

if (-not $SkipInstall) {
    & $python -m pip install --requirement requirements-build.txt

    if ($LASTEXITCODE -ne 0) {
        throw "Build dependency installation failed."
    }
}

& $python (Join-Path $PSScriptRoot "prepare_embedding_model.py")

if ($LASTEXITCODE -ne 0) {
    throw "Embedding model preparation failed."
}

& $python -m PyInstaller --noconfirm --clean $spec

if ($LASTEXITCODE -ne 0) {
    throw "EXE build failed."
}

$executable = Join-Path $distDirectory "KnowledgeAssistant.exe"
$startGuide = Join-Path $projectRoot "START_TUTAJ.txt"

if (-not (Test-Path -LiteralPath $executable)) {
    throw "Built executable not found: $executable"
}

Copy-Item -LiteralPath $startGuide -Destination $distDirectory -Force

New-Item -ItemType Directory -Force -Path $releaseDirectory | Out-Null

Write-Host ""
Write-Host "BUILD WINDOWS - READY"
Write-Host "EXE: $executable"

if (-not $SkipArchive) {
    Compress-Archive -Path (Join-Path $distDirectory "*") -DestinationPath $archive -Force
    $sizeMegabytes = (Get-Item -LiteralPath $archive).Length / 1MB
    Write-Host ("ZIP: {0} ({1:N1} MB)" -f $archive, $sizeMegabytes)
}
