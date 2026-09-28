param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$DescriptionColumn,

    [Parameter(Mandatory = $true)]
    [string]$Instruction,

    [string]$Worksheet,
    [int]$HeaderRow = 1,
    [int]$MaxTickets = 2000,
    [int]$BatchSize = 50,
    [ValidateRange(1, 256)]
    [int]$TorchThreads = 4,
    [ValidateRange(1, 4096)]
    [int]$LayaBatchSize = 16,
    [double]$Threshold = 0.5,
    [ValidateSet("auto", "english", "multilingual", "typed-decisions")]
    [string]$Model = "auto",
    [ValidateSet("cpu", "openvino")]
    [string]$Engine = "cpu",
    [ValidateSet("CPU", "GPU", "AUTO")]
    [string]$OpenVinoDevice = "AUTO"
)

$ErrorActionPreference = "Stop"
$executionStart = Get-Date
$executionTimer = [System.Diagnostics.Stopwatch]::StartNew()

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "This script requires PowerShell 7 or later for local process invocation."
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input file not found: $InputPath"
}

if ($BatchSize -lt 1) {
    throw "BatchSize must be at least 1."
}

if ($LayaBatchSize -lt 1) {
    throw "LayaBatchSize must be at least 1."
}

if ($MaxTickets -lt 1) {
    throw "MaxTickets must be at least 1."
}

if ($Threshold -lt 0 -or $Threshold -gt 1) {
    throw "Threshold must be between 0 and 1."
}

if ($Engine -eq "openvino" -and $Model -ne "auto") {
    throw "The openvino engine only supports the bundled English checkpoint; do not pass -Model with -Engine openvino."
}

# -----------------------------------------------------------------------------
# Read the Excel file through the local Excel COM API.
# The input workbook is opened read-only and is never modified.
# -----------------------------------------------------------------------------
$excel = $null
$workbook = $null
$sheet = $null

try {
    $resolvedPath = (Resolve-Path -LiteralPath $InputPath).Path

    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false

    $workbook = $excel.Workbooks.Open($resolvedPath, 0, $true)

    if ($Worksheet) {
        $sheet = $workbook.Worksheets.Item($Worksheet)
    }
    else {
        $sheet = $workbook.Worksheets.Item(1)
    }

    $usedRange = $sheet.UsedRange
    $firstColumn = $usedRange.Column
    $lastColumn = $usedRange.Column + $usedRange.Columns.Count - 1
    $lastRow = $usedRange.Row + $usedRange.Rows.Count - 1

    $descriptionColumnIndex = $null

    for ($column = $firstColumn; $column -le $lastColumn; $column++) {
        $header = [string]$sheet.Cells.Item($HeaderRow, $column).Value2

        if ($header.Trim() -eq $DescriptionColumn.Trim()) {
            $descriptionColumnIndex = $column
            break
        }
    }

    if ($null -eq $descriptionColumnIndex) {
        throw "Column '$DescriptionColumn' was not found on row $HeaderRow."
    }

    $tickets = [System.Collections.Generic.List[object]]::new()

    for ($row = $HeaderRow + 1; $row -le $lastRow; $row++) {
        if ($tickets.Count -ge $MaxTickets) {
            break
        }

        $description = [string]$sheet.Cells.Item($row, $descriptionColumnIndex).Value2

        if ([string]::IsNullOrWhiteSpace($description)) {
            continue
        }

        $tickets.Add([PSCustomObject]@{
            RowNumber   = $row
            Description = $description
        })
    }
}
finally {
    if ($workbook) {
        $workbook.Close($false)
    }

    if ($excel) {
        $excel.Quit()
    }

    if ($sheet) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($sheet)
    }

    if ($workbook) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook)
    }

    if ($excel) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
    }

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

if ($tickets.Count -eq 0) {
    throw "No non-empty tickets found in column '$DescriptionColumn'."
}

$runnerPath = if ($Engine -eq "openvino") {
    Join-Path $PSScriptRoot "openvino_runner.py"
}
else {
    Join-Path $PSScriptRoot "laya_runner.py"
}
if (-not (Test-Path -LiteralPath $runnerPath)) {
    throw "Runner script not found: $runnerPath"
}

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python environment not found. Run 'uv sync --python 3.11' from $PSScriptRoot first."
}

$cacheRoot = Join-Path $PSScriptRoot ".laya-cache"
$payload = @{
    batchSize   = $BatchSize
    instruction = $Instruction
    model       = $Model
    tickets     = @(
        $tickets | ForEach-Object {
            @{
                rowNumber   = $_.RowNumber
                description = $_.Description
            }
        }
    )
} | ConvertTo-Json -Depth 10

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $pythonPath
$startInfo.WorkingDirectory = $PSScriptRoot
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardInput = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.StandardInputEncoding = [System.Text.UTF8Encoding]::new($false)
$startInfo.StandardOutputEncoding = [System.Text.UTF8Encoding]::new($false)
$startInfo.Environment["PYTHONUTF8"] = "1"
$startInfo.Environment["HF_HOME"] = $cacheRoot
$startInfo.Environment["HF_HUB_CACHE"] = Join-Path $cacheRoot "hub"
$startInfo.Environment["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if ($Engine -eq "openvino") {
    $checkpointRoot = Join-Path $cacheRoot "checkpoints\mattepiu-laya-onnx-a7f385b"
    $onnxPath = Join-Path $checkpointRoot "fp16_onlygpu_unverified\laya_fp16.onnx"
    if (-not (Test-Path -LiteralPath $onnxPath)) {
        throw "OpenVINO checkpoint not found: $onnxPath"
    }
    $startInfo.Environment["LAYA_OV_ONNX_PATH"] = $onnxPath
    $startInfo.Environment["LAYA_OV_TOKENIZER_DIR"] = $checkpointRoot
    $startInfo.Environment["LAYA_OV_DEVICE"] = $OpenVinoDevice
    $startInfo.Environment["LAYA_OV_BATCH_SIZE"] = [string]$LayaBatchSize
    $startInfo.Environment["LAYA_OV_CACHE_DIR"] = Join-Path $cacheRoot "openvino-cache"
}
else {
    $startInfo.Environment["LAYA_TORCH_THREADS"] = [string]$TorchThreads
    $startInfo.Environment["LAYA_BATCH_SIZE"] = [string]$LayaBatchSize
}
$startInfo.ArgumentList.Add($runnerPath)

Write-Host "Tickets to evaluate: $($tickets.Count)"
Write-Host "Batch size:          $BatchSize"
if ($Engine -eq "openvino") {
    Write-Host "Model:               english (fp16 onnx, fixed)"
    Write-Host "Runtime:             OpenVINO ($OpenVinoDevice) via .venv"
}
else {
    Write-Host "Model:               $Model"
    Write-Host "Runtime:             local CPU via .venv"
}
Write-Host "Weights cache:       $cacheRoot"
Write-Host "Threshold:           $Threshold"
Write-Host ""
Write-Host "Starting local Laya inference. First run downloads the model weights into the repository."

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
try {
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $process.StandardInput.Write($payload)
    $process.StandardInput.Close()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()

    if ($process.ExitCode -ne 0) {
        throw "Local Laya inference failed (uv exit code $($process.ExitCode)). See the uv/Python output above."
    }

    $response = ConvertFrom-Json -InputObject $stdout
    if ($response.results.Count -ne $tickets.Count) {
        throw "Laya returned $($response.results.Count) results for $($tickets.Count) tickets."
    }
    if ($null -eq $response.inference) {
        throw "Laya did not return inference timing data."
    }
}
finally {
    $process.Dispose()
}

$allResults = [System.Collections.Generic.List[object]]::new()
foreach ($result in $response.results) {
    $score = $null
    $automatable = $null
    if ($null -eq $result.error) {
        $score = [double]$result.score
        $automatable = ($score -ge $Threshold)
    }

    $allResults.Add([PSCustomObject]@{
        RowNumber   = [int]$result.rowNumber
        Score       = $score
        Automatable = $automatable
        Error       = $result.error
    })
}

$successful = @($allResults | Where-Object { $null -eq $_.Error })
$failed = @($allResults | Where-Object { $null -ne $_.Error })
$automatable = @($successful | Where-Object { $_.Automatable -eq $true })
$notAutomatable = @($successful | Where-Object { $_.Automatable -eq $false })
$executionTimer.Stop()
$executionEnd = Get-Date
$inferenceDuration = [TimeSpan]::FromSeconds([double]$response.inference.seconds)

Write-Host ""
Write-Host "==================== RESULT ===================="
Write-Host "Started:                  $($executionStart.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Finished:                 $($executionEnd.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Execution time:           $($executionTimer.Elapsed.ToString('c'))"
Write-Host "Inference started:        $($response.inference.startedAt)"
Write-Host "Inference finished:       $($response.inference.finishedAt)"
Write-Host "Inference time:           $($inferenceDuration.ToString('c'))"
Write-Host "Checkpoint(s):             $($response.inference.checkpoints -join ', ')"
Write-Host "Tickets evaluated:       $($allResults.Count)"
Write-Host "Successful evaluations:  $($successful.Count)"
Write-Host "Automatable:              $($automatable.Count)"
Write-Host "Not automatable:          $($notAutomatable.Count)"
Write-Host "Errors:                   $($failed.Count)"

if ($successful.Count -gt 0) {
    $automatablePercent = [Math]::Round(($automatable.Count / $successful.Count) * 100, 2)
    $notAutomatablePercent = [Math]::Round(($notAutomatable.Count / $successful.Count) * 100, 2)

    Write-Host "Automatable %:            $automatablePercent%"
    Write-Host "Not automatable %:        $notAutomatablePercent%"
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed rows:"
    $failed | Select-Object RowNumber, Error | Format-Table -AutoSize
}
