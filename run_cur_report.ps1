param(
    [string]$CsvPath,
    [string]$OutputPath = "MyProduct/report_data.json"
)

# Prompt for CSV path if not provided
if (-not $CsvPath -or $CsvPath.Trim().Length -eq 0) {
    $CsvPath = Read-Host "Enter path to CUR CSV file"
}

if (-not (Test-Path $CsvPath)) {
    Write-Error "CSV file not found: $CsvPath"
    exit 1
}

# Resolve absolute paths
$CsvFull = (Resolve-Path $CsvPath).Path

$outputDir = Split-Path $OutputPath -Parent
if (-not $outputDir -or $outputDir.Trim().Length -eq 0) { $outputDir = "." }
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
$OutputFull = (Resolve-Path -Path $outputDir).Path
$OutputFull = Join-Path $OutputFull (Split-Path $OutputPath -Leaf)

Write-Host "Input CSV:  $CsvFull"
Write-Host "Output JSON: $OutputFull"

# Run the summarizer
python cur_to_json.py --input "$CsvFull" --output "$OutputFull"

if ($LASTEXITCODE -ne 0) {
    Write-Error "cur_to_json.py failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Done. Load report.html (it reads MyProduct/report_data.json by default)." -ForegroundColor Green

# Auto-open the report HTML
$reportPath = Join-Path (Split-Path $OutputFull -Parent) "report.html"
if (-not (Test-Path $reportPath)) {
    # Fall back to known location if output is elsewhere
    $reportPath = "MyProduct/report.html"
}
if (Test-Path $reportPath) {
    Write-Host "Opening report via http://localhost:8000/report.html ..."
    $reportDir = Split-Path $reportPath -Parent
    $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) {
        Start-Process -FilePath python -ArgumentList "-m","http.server","8000" -WorkingDirectory $reportDir | Out-Null
        Start-Sleep -Seconds 1
    }
    Start-Process "http://localhost:8000/report.html"
} else {
    Write-Warning "Report HTML not found to open automatically."
}
