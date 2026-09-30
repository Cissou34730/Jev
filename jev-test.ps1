param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [ValidateSet("excel", "json")]
    [string]$InputFormat = "excel",

    [string]$DescriptionColumn,

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
    [string]$OpenVinoDevice = "AUTO",
    [int]$OpenVinoMaxLen = 128,
    [switch]$OpenVinoAutoMaxLen,
    [ValidateRange(0, 100)]
    [double]$OpenVinoMaxLenTruncatePercent = 0,
    [switch]$OpenVinoForceFp32
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
    throw "The openvino engine only supports the bundled multilingual master checkpoint; do not pass -Model with -Engine openvino."
}

if ($Engine -ne "openvino" -and ($PSBoundParameters.ContainsKey("OpenVinoMaxLen") -or $OpenVinoAutoMaxLen -or $OpenVinoMaxLenTruncatePercent -gt 0 -or $OpenVinoForceFp32)) {
    throw "-OpenVinoMaxLen/-OpenVinoAutoMaxLen/-OpenVinoMaxLenTruncatePercent/-OpenVinoForceFp32 only apply to -Engine openvino."
}

if ($OpenVinoMaxLen -lt 1) {
    throw "OpenVinoMaxLen must be at least 1."
}

if ($OpenVinoMaxLenTruncatePercent -gt 0 -and -not $OpenVinoAutoMaxLen) {
    throw "-OpenVinoMaxLenTruncatePercent only applies together with -OpenVinoAutoMaxLen."
}

if ($InputFormat -eq "excel" -and -not $DescriptionColumn) {
    throw "-DescriptionColumn is required when -InputFormat excel."
}

# Only -Engine openvino combined with -InputFormat json answers each ticket against the JSON
# file's own multi-question schema ("questions": {id -> {type, instructions, criteria}}).
# Every other combination keeps the legacy single hardcoded noul "automatable" question built
# from -Instruction.
$useQuestionsFromJson = ($Engine -eq "openvino" -and $InputFormat -eq "json")

if (-not $useQuestionsFromJson -and [string]::IsNullOrWhiteSpace($Instruction)) {
    throw "-Instruction is required unless -Engine openvino is combined with -InputFormat json (in that case the question schema comes from the JSON file's own 'questions' object)."
}

if ($InputFormat -eq "json") {
    # -------------------------------------------------------------------------
    # Read tickets from a JSON file: a top-level "tickets" array where each
    # entry has a "state" string field used as the ticket description, and (for
    # -Engine openvino only) a top-level "questions" object providing the
    # multi-question schema each ticket is answered against. The "expected"
    # per-ticket field is ignored (no automatic scoring against it).
    # -------------------------------------------------------------------------
    $resolvedPath = (Resolve-Path -LiteralPath $InputPath).Path
    $jsonContent = Get-Content -LiteralPath $resolvedPath -Raw | ConvertFrom-Json

    if ($null -eq $jsonContent.tickets) {
        throw "JSON input must have a top-level 'tickets' array (with a 'state' string field per entry). File: $resolvedPath"
    }

    $ticketQuestions = $null
    $questionIds = @()
    if ($useQuestionsFromJson) {
        if ($null -eq $jsonContent.questions -or @($jsonContent.questions.PSObject.Properties).Count -eq 0) {
            throw "JSON input must have a non-empty top-level 'questions' object (one entry per question id) when using -Engine openvino with -InputFormat json. File: $resolvedPath"
        }
        $ticketQuestions = $jsonContent.questions
        $questionIds = @($ticketQuestions.PSObject.Properties.Name)
    }

    $tickets = [System.Collections.Generic.List[object]]::new()
    $index = 0

    foreach ($item in $jsonContent.tickets) {
        if ($tickets.Count -ge $MaxTickets) {
            break
        }

        $index++

        if ($null -eq $item.state -or [string]::IsNullOrWhiteSpace([string]$item.state)) {
            continue
        }

        $tickets.Add([PSCustomObject]@{
            RowNumber   = $index
            Description = [string]$item.state
        })
    }
}
else {
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
}

if ($tickets.Count -eq 0) {
    if ($InputFormat -eq "json") {
        throw "No non-empty tickets found (looked for a 'state' string field on each entry of the 'tickets' array)."
    }
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
$payloadTable = [ordered]@{
    batchSize = $BatchSize
    model     = $Model
    tickets   = @(
        $tickets | ForEach-Object {
            @{
                rowNumber   = $_.RowNumber
                description = $_.Description
            }
        }
    )
}
if ($useQuestionsFromJson) {
    $payloadTable["questions"] = $ticketQuestions
}
else {
    $payloadTable["instruction"] = $Instruction
}
$payload = $payloadTable | ConvertTo-Json -Depth 10

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
    $checkpointRoot = Join-Path $cacheRoot "checkpoints\laya-master"
    $onnxPath = Join-Path $checkpointRoot "laya.onnx"
    if (-not (Test-Path -LiteralPath $onnxPath)) {
        throw "OpenVINO checkpoint not found: $onnxPath. Run '.venv\Scripts\python.exe export_laya_master.py' first."
    }
    $startInfo.Environment["LAYA_OV_ONNX_PATH"] = $onnxPath
    $startInfo.Environment["LAYA_OV_TOKENIZER_DIR"] = $checkpointRoot
    $startInfo.Environment["LAYA_OV_DEVICE"] = $OpenVinoDevice
    $startInfo.Environment["LAYA_OV_BATCH_SIZE"] = [string]$LayaBatchSize
    $startInfo.Environment["LAYA_OV_CACHE_DIR"] = Join-Path $cacheRoot "openvino-cache"
    $startInfo.Environment["LAYA_OV_MAX_LEN"] = [string]$OpenVinoMaxLen
    $startInfo.Environment["LAYA_OV_AUTO_MAX_LEN"] = if ($OpenVinoAutoMaxLen) { "1" } else { "0" }
    $startInfo.Environment["LAYA_OV_MAX_LEN_TRUNCATE_PERCENT"] = [string]$OpenVinoMaxLenTruncatePercent
    $startInfo.Environment["LAYA_OV_FORCE_FP32"] = if ($OpenVinoForceFp32) { "1" } else { "0" }
}
else {
    $startInfo.Environment["LAYA_TORCH_THREADS"] = [string]$TorchThreads
    $startInfo.Environment["LAYA_BATCH_SIZE"] = [string]$LayaBatchSize
}
$startInfo.ArgumentList.Add($runnerPath)

Write-Host "Tickets to evaluate: $($tickets.Count)"
if ($Engine -eq "openvino") {
    Write-Host "Batch size:          $LayaBatchSize tickets per forward pass (OpenVINO; -BatchSize is ignored by this engine)"
    if ($useQuestionsFromJson) {
        $rowsPerBatch = $LayaBatchSize * $questionIds.Count
        Write-Host "Questions:           $($questionIds -join ', ') (from JSON; $($questionIds.Count) question(s)/ticket -> $rowsPerBatch model rows per forward pass)"
    }
    Write-Host "Model:               convaiinnovations/laya-multilingual (laya-master, fp32 onnx, dynamic n/L)"
    Write-Host "Runtime:             OpenVINO ($OpenVinoDevice) via .venv"
    if ($OpenVinoAutoMaxLen) {
        if ($OpenVinoMaxLenTruncatePercent -gt 0) {
            Write-Host "Max len:             auto, accepting $OpenVinoMaxLenTruncatePercent% of tickets truncated (measured from these tickets at runtime; -OpenVinoMaxLen ignored)"
        }
        else {
            Write-Host "Max len:             auto (measured from these tickets at runtime; -OpenVinoMaxLen ignored)"
        }
    }
    else {
        Write-Host "Max len:             $OpenVinoMaxLen (fixed; longer descriptions get truncated)"
    }
    if ($OpenVinoForceFp32) {
        Write-Host "Precision:           forced fp32 (GPU defaults to fp16 otherwise; slower, more accurate)"
    }
    else {
        Write-Host "Precision:           device default (fp16 on GPU, fp32/bf16 on CPU)"
    }
}
else {
    Write-Host "Batch size:          $BatchSize"
    Write-Host "Model:               $Model"
    Write-Host "Runtime:             local CPU via .venv"
}
Write-Host "Weights cache:       $cacheRoot"
Write-Host "Threshold:           $Threshold"
Write-Host ""
if ($Engine -eq "openvino") {
    Write-Host "Starting local OpenVINO inference. The checkpoint is already local; no download happens. The first run for a given -LayaBatchSize compiles the model (slow), every run after that only reloads it onto the device (still several seconds, this is not a bug)."
}
else {
    Write-Host "Starting local Laya inference. First run downloads the model weights into the repository."
}

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
if ($useQuestionsFromJson) {
    foreach ($result in $response.results) {
        $row = [ordered]@{
            RowNumber = [int]$result.rowNumber
            Error     = $result.error
        }
        foreach ($qid in $questionIds) {
            $row[$qid] = if ($null -eq $result.error) { $result.answers.$qid } else { $null }
        }
        $allResults.Add([PSCustomObject]$row)
    }
}
else {
    foreach ($result in $response.results) {
        $score = $null
        $automatable = $null
        if ($null -eq $result.error) {
            $score = [double]$result.answers.automatable.noul
            $automatable = ($score -ge $Threshold)
        }

        $allResults.Add([PSCustomObject]@{
            RowNumber   = [int]$result.rowNumber
            Score       = $score
            Automatable = $automatable
            Error       = $result.error
        })
    }
}

$successful = @($allResults | Where-Object { $null -eq $_.Error })
$failed = @($allResults | Where-Object { $null -ne $_.Error })
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
if ($Engine -eq "openvino") {
    Write-Host "Compile time:             $([Math]::Round([double]$response.inference.compileSeconds, 2))s ($($response.inference.cacheStatus))"
    if ($response.inference.maxLenAuto) {
        Write-Host "Max len used:             $($response.inference.maxLen) (auto; $($response.inference.maxLenTruncatedTickets) ticket(s), $([Math]::Round([double]$response.inference.maxLenTruncatedPercent, 1))%, had their description truncated)"
    }
    else {
        Write-Host "Max len used:             $($response.inference.maxLen) (fixed)"
    }
}
Write-Host "Tickets evaluated:       $($allResults.Count)"
Write-Host "Successful evaluations:  $($successful.Count)"
Write-Host "Errors:                   $($failed.Count)"

if ($useQuestionsFromJson) {
    foreach ($qid in $questionIds) {
        $qtype = $ticketQuestions.$qid.type
        Write-Host ""
        Write-Host "---- Question '$qid' ($qtype) ----"
        switch ($qtype) {
            "noul" {
                $values = @($successful | ForEach-Object { [double]$_.$qid.noul })
                $trueCount = @($values | Where-Object { $_ -ge $Threshold }).Count
                $falseCount = $values.Count - $trueCount
                Write-Host "True (>= $Threshold):     $trueCount"
                Write-Host "False (< $Threshold):     $falseCount"
            }
            "choice" {
                $successful | ForEach-Object { $_.$qid.choice } | Group-Object | Sort-Object Count -Descending | ForEach-Object {
                    Write-Host "$($_.Name):$(' ' * [Math]::Max(1, 20 - $_.Name.Length))$($_.Count)"
                }
            }
            "score" {
                $values = @($successful | ForEach-Object { [double]$_.$qid.score })
                if ($values.Count -gt 0) {
                    $avg = [Math]::Round(($values | Measure-Object -Average).Average, 3)
                    Write-Host "Average score:           $avg"
                }
                $levels = @($ticketQuestions.$qid.criteria)
                $successful | ForEach-Object { [int][Math]::Round($_.$qid.score) } | Group-Object | Sort-Object Name | ForEach-Object {
                    $levelIndex = [int]$_.Name
                    $label = if ($levelIndex -ge 0 -and $levelIndex -lt $levels.Count) { "$levelIndex ($($levels[$levelIndex]))" } else { "$levelIndex" }
                    Write-Host "$($label):$(' ' * [Math]::Max(1, 28 - $label.Length))$($_.Count)"
                }
            }
        }
    }
}
else {
    $automatable = @($successful | Where-Object { $_.Automatable -eq $true })
    $notAutomatable = @($successful | Where-Object { $_.Automatable -eq $false })

    Write-Host "Automatable:              $($automatable.Count)"
    Write-Host "Not automatable:          $($notAutomatable.Count)"

    if ($successful.Count -gt 0) {
        $automatablePercent = [Math]::Round(($automatable.Count / $successful.Count) * 100, 2)
        $notAutomatablePercent = [Math]::Round(($notAutomatable.Count / $successful.Count) * 100, 2)

        Write-Host "Automatable %:            $automatablePercent%"
        Write-Host "Not automatable %:        $notAutomatablePercent%"
    }
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed rows:"
    $failed | Select-Object RowNumber, Error | Format-Table -AutoSize
}
