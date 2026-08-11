param(
    [switch]$SkipInstall,
    [switch]$SkipWindowsBuild,
    [string]$OllamaVersion = "v0.32.8",
    [string]$ModelName = "gemma3:4b",
    [string]$SourceModelsDirectory = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$cacheDirectory = Join-Path $PSScriptRoot "cache"
$releaseDirectory = Join-Path $projectRoot "release"
$standardDist = Join-Path $projectRoot "dist\KnowledgeAssistant"
$offlineDist = Join-Path $projectRoot "dist\KnowledgeAssistant-Full-Offline"
$version = (Get-Content -Raw (Join-Path $projectRoot "VERSION")).Trim()
$archiveName = "KnowledgeAssistant-Full-Offline-Windows-x64-v$version.zip"
$archive = Join-Path $releaseDirectory $archiveName
$partsDirectory = Join-Path $releaseDirectory "KnowledgeAssistant-Full-Offline-v$version-parts"
$ollamaArchive = Join-Path $cacheDirectory "ollama-windows-amd64-$OllamaVersion.zip"
$ollamaExtracted = Join-Path $cacheDirectory "ollama-windows-amd64-$OllamaVersion"
$ollamaExpectedHash = "F8C2CF97739FD940961AE101C79C60E5A5DF6596E2354565E902059DCF12BBB0"
$gemmaTerms = Join-Path $cacheDirectory "GEMMA_TERMS_OF_USE.html"

if ([string]::IsNullOrWhiteSpace($SourceModelsDirectory)) {
    $SourceModelsDirectory = Join-Path $env:USERPROFILE ".ollama\models"
}

function Remove-ProjectBuildTarget {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $projectPrefix = $projectRoot + [IO.Path]::DirectorySeparatorChar

    if (-not $resolved.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the project: $resolved"
    }

    Remove-Item -LiteralPath $resolved -Recurse -Force
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing virtual environment: $python"
}

if (-not (Test-Path -LiteralPath $SourceModelsDirectory)) {
    throw "Missing Ollama model directory: $SourceModelsDirectory"
}

New-Item -ItemType Directory -Force -Path $cacheDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $releaseDirectory | Out-Null
Set-Location -LiteralPath $projectRoot

if (-not $SkipWindowsBuild) {
    $windowsBuild = Join-Path $PSScriptRoot "build_windows.ps1"

    if ($SkipInstall) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $windowsBuild -SkipInstall -SkipArchive
    }
    else {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $windowsBuild -SkipArchive
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Base Windows build failed."
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $standardDist "KnowledgeAssistant.exe"))) {
    throw "Base Windows build is missing: $standardDist"
}

if (-not (Test-Path -LiteralPath $ollamaArchive)) {
    gh release download $OllamaVersion --repo "ollama/ollama" --pattern "ollama-windows-amd64.zip" --dir $cacheDirectory

    if ($LASTEXITCODE -ne 0) {
        throw "Ollama download failed."
    }

    Move-Item -LiteralPath (Join-Path $cacheDirectory "ollama-windows-amd64.zip") -Destination $ollamaArchive
}

$ollamaActualHash = (Get-FileHash -LiteralPath $ollamaArchive -Algorithm SHA256).Hash

if ($ollamaActualHash -ne $ollamaExpectedHash) {
    throw "Ollama archive checksum mismatch."
}

if (-not (Test-Path -LiteralPath (Join-Path $ollamaExtracted "ollama.exe"))) {
    Remove-ProjectBuildTarget -Path $ollamaExtracted
    New-Item -ItemType Directory -Force -Path $ollamaExtracted | Out-Null
    & tar.exe -xf $ollamaArchive -C $ollamaExtracted

    if ($LASTEXITCODE -ne 0) {
        throw "Ollama extraction failed."
    }
}

if (-not (Test-Path -LiteralPath $gemmaTerms)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://ai.google.dev/gemma/terms" -OutFile $gemmaTerms
}

Remove-ProjectBuildTarget -Path $offlineDist
New-Item -ItemType Directory -Force -Path $offlineDist | Out-Null
Copy-Item -Path (Join-Path $standardDist "*") -Destination $offlineDist -Recurse -Force

$offlineRoot = Join-Path $offlineDist "_internal\offline"
$offlineOllama = Join-Path $offlineRoot "ollama"
$offlineModels = Join-Path $offlineRoot "models"
$offlineLicenses = Join-Path $offlineRoot "licenses"
New-Item -ItemType Directory -Force -Path $offlineOllama | Out-Null
New-Item -ItemType Directory -Force -Path $offlineLicenses | Out-Null
Copy-Item -Path (Join-Path $ollamaExtracted "*") -Destination $offlineOllama -Recurse -Force

& $python (Join-Path $PSScriptRoot "prepare_offline_ollama.py") --source-models $SourceModelsDirectory --destination-models $offlineModels --model $ModelName

if ($LASTEXITCODE -ne 0) {
    throw "Offline model export failed."
}

$assetsDirectory = Join-Path $PSScriptRoot "assets"
Copy-Item -LiteralPath (Join-Path $assetsDirectory "FULL_OFFLINE") -Destination $offlineRoot -Force
Copy-Item -LiteralPath (Join-Path $assetsDirectory "OLLAMA_LICENSE.txt") -Destination $offlineLicenses -Force
Copy-Item -LiteralPath (Join-Path $assetsDirectory "GEMMA_NOTICE.txt") -Destination $offlineLicenses -Force
Copy-Item -LiteralPath $gemmaTerms -Destination $offlineLicenses -Force
Copy-Item -LiteralPath (Join-Path $projectRoot "START_TUTAJ_OFFLINE.txt") -Destination (Join-Path $offlineDist "START_TUTAJ.txt") -Force

if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}

& tar.exe -a -cf $archive -C $offlineDist .

if ($LASTEXITCODE -ne 0) {
    throw "Full Offline ZIP creation failed."
}

Remove-ProjectBuildTarget -Path $partsDirectory
New-Item -ItemType Directory -Force -Path $partsDirectory | Out-Null

& $python (Join-Path $PSScriptRoot "split_release.py") --source $archive --output-directory $partsDirectory --part-size-mib 1900

if ($LASTEXITCODE -ne 0) {
    throw "Release split failed."
}

Copy-Item -LiteralPath (Join-Path $assetsDirectory "SCAL_I_ROZPAKUJ.cmd") -Destination $partsDirectory -Force
Copy-Item -LiteralPath (Join-Path $assetsDirectory "SCAL_I_ROZPAKUJ.ps1") -Destination $partsDirectory -Force
Copy-Item -LiteralPath (Join-Path $projectRoot "START_TUTAJ_OFFLINE.txt") -Destination (Join-Path $partsDirectory "INSTRUKCJA.txt") -Force

$archiveSize = (Get-Item -LiteralPath $archive).Length / 1GB
$distSize = (Get-ChildItem -LiteralPath $offlineDist -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
$partCount = (Get-ChildItem -LiteralPath $partsDirectory -Filter "*.part*" -File).Count

Write-Host ""
Write-Host "BUILD FULL OFFLINE - READY"
Write-Host ("DIST: {0} ({1:N2} GB)" -f $offlineDist, $distSize)
Write-Host ("ZIP: {0} ({1:N2} GB)" -f $archive, $archiveSize)
Write-Host ("GITHUB PARTS: {0} ({1} files)" -f $partsDirectory, $partCount)
