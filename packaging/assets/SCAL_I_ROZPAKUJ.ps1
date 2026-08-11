$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$archiveName = (Get-Content -Raw (Join-Path $root "FULL_ARCHIVE_NAME.txt")).Trim()
$expectedArchiveHash = (Get-Content -Raw (Join-Path $root "FULL_ARCHIVE_SHA256.txt")).Trim()
$partHashes = Get-Content -Raw (Join-Path $root "PARTS_SHA256.json") | ConvertFrom-Json
$parts = Get-ChildItem -LiteralPath $root -Filter "$archiveName.part*" -File | Sort-Object Name

if ($parts.Count -eq 0) {
    throw "Nie znaleziono czesci archiwum. Pobierz wszystkie pliki partXXX."
}

foreach ($part in $parts) {
    $property = $partHashes.PSObject.Properties[$part.Name]

    if ($null -eq $property) {
        throw "Brak sumy kontrolnej dla $($part.Name)."
    }

    $actualPartHash = (Get-FileHash -LiteralPath $part.FullName -Algorithm SHA256).Hash

    if ($actualPartHash -ne $property.Value) {
        throw "Plik $($part.Name) jest uszkodzony. Pobierz go ponownie."
    }
}

$archivePath = Join-Path $root $archiveName
$archiveStream = [IO.File]::Open($archivePath, [IO.FileMode]::Create, [IO.FileAccess]::Write)

try {
    foreach ($part in $parts) {
        $partStream = [IO.File]::OpenRead($part.FullName)

        try {
            $partStream.CopyTo($archiveStream, 8MB)
        }
        finally {
            $partStream.Dispose()
        }
    }
}
finally {
    $archiveStream.Dispose()
}

$actualArchiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash

if ($actualArchiveHash -ne $expectedArchiveHash) {
    Remove-Item -LiteralPath $archivePath -Force
    throw "Scalone archiwum ma nieprawidlowa sume kontrolna."
}

$destinationName = [IO.Path]::GetFileNameWithoutExtension($archiveName)
$destination = Join-Path $root $destinationName

if (Test-Path -LiteralPath $destination) {
    throw "Folder docelowy juz istnieje: $destination"
}

New-Item -ItemType Directory -Path $destination | Out-Null
& tar.exe -xf $archivePath -C $destination

if ($LASTEXITCODE -ne 0) {
    throw "Rozpakowanie archiwum nie powiodlo sie."
}

Remove-Item -LiteralPath $archivePath -Force
Write-Host ""
Write-Host "Knowledge Assistant Full Offline zostal rozpakowany do:"
Write-Host $destination
Write-Host ""
Write-Host "Uruchom KnowledgeAssistant.exe z tego folderu."
