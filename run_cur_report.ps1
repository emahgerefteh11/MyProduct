param(
    [string[]]$CsvPath,
    [string]$OutputPath = "MyProduct/report_data.json",
    [string]$NamingPath,
    [string]$NamingMode,
    [string[]]$Label
)

# Prompt for CSV path if not provided
if (-not $CsvPath -or $CsvPath.Count -eq 0) {
    $CsvPath = @(Read-Host "Enter path to CUR CSV file")
}

$CsvPath = @($CsvPath)
if ($CsvPath.Count -eq 1 -and $CsvPath[0] -like '*,*') {
    $CsvPath = $CsvPath[0].Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

$Label = @($Label)
if ($Label.Count -eq 1 -and $Label[0] -like '*,*') {
    $Label = $Label[0].Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

$CsvFull = @()
foreach ($csv in $CsvPath) {
    if (-not (Test-Path $csv)) {
        Write-Error "CSV file not found: $csv"
        exit 1
    }
    $CsvFull += (Resolve-Path $csv).Path
}

$outputDir = Split-Path $OutputPath -Parent
if (-not $outputDir -or $outputDir.Trim().Length -eq 0) { $outputDir = "." }
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
$OutputFull = (Resolve-Path -Path $outputDir).Path
$OutputFull = Join-Path $OutputFull (Split-Path $OutputPath -Leaf)

Write-Host "Input CSV(s):"
$CsvFull | ForEach-Object { Write-Host "  $_" }
Write-Host "Output JSON: $OutputFull"

# Run the summarizer
$pythonArgs = @("cur_to_json.py", "--output", "$OutputFull")
foreach ($csv in $CsvFull) {
    $pythonArgs += @("--input", "$csv")
}
if ($Label -and $Label.Count -gt 0) {
    foreach ($name in $Label) {
        $pythonArgs += @("--label", "$name")
    }
    Write-Host "Labels: $($Label -join ', ')"
}
if ($NamingPath -and $NamingPath.Trim().Length -gt 0) {
    $pythonArgs += @("--naming", "$NamingPath")
    Write-Host "Naming config: $NamingPath"
}
if ($NamingMode -and $NamingMode.Trim().Length -gt 0) {
    $pythonArgs += @("--naming-mode", "$NamingMode")
    Write-Host "Naming mode: $NamingMode"
}
python @pythonArgs

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
