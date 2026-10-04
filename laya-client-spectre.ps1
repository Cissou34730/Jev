#Requires -Version 7.0
<#
.SYNOPSIS
Same client as laya-client.ps1, with the live table drawn by PwshSpectreConsole.

.DESCRIPTION
Everything about the requests is identical to laya-client.ps1: the same chunking by -BatchSize, the
same one request per chunk with batch_size asking for a single forward pass, the same silent wait on
503 honouring Retry-After, and the same time measured from the moment the server accepted a request to
the moment the answer came back. Only the drawing differs, and it is the library that draws it.

It duplicates laya-client.ps1 on purpose, so trying the library cannot break the client that works.
Delete whichever loses.

Requires the module: Install-Module PwshSpectreConsole -Scope CurrentUser
Requires SpectreScriptRenderable.ps1 next to this file: the live display needs a renderable that
rebuilds itself on every draw, and a class can only be parsed once the module is loaded.

The input is either {"questions": {...}, "tickets": [...]} or a bare array of tickets (then use
-QuestionsPath). -Model defaults to "multilingual", a name laya-serve routes.

-NoulThreshold is where a noul answer flips to true (default 0.5). -ScoreLowCut and -ScoreHighCut
replace the argmax on a 3-level score question with two boundaries on the score number, e.g.
-ScoreLowCut 0.95 -ScoreHighCut 1.25. Both change only how an answer is read, never the model, and
both default to the behaviour above.

.EXAMPLE
.\laya-client-spectre.ps1 -InputPath .\tickets.json
.\laya-client-spectre.ps1 -InputPath .\tickets.json -BatchSize 50 -MaxWaitSeconds 600
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$InputPath,
    [string]$QuestionsPath,
    [string]$TextProperty = 'state',
    [string]$IdProperty = 'id',
    [string]$BaseUrl = 'http://127.0.0.1:8001',
    [int]$MaxTickets = 0,
    [string]$Model = 'multilingual',
    [int]$BatchSize = 50,
    [int]$MaxWaitSeconds = 300,
    [int]$RefreshMs = 200,
    [int]$TimeoutSeconds = 3600,
    [string]$OutputPath,
    [double]$NoulThreshold = 0.5,
    [double]$ScoreLowCut,
    [double]$ScoreHighCut
)

$ErrorActionPreference = 'Stop'

if ($PSBoundParameters.ContainsKey('ScoreLowCut') -ne $PSBoundParameters.ContainsKey('ScoreHighCut')) {
    throw '-ScoreLowCut and -ScoreHighCut go together: pass both, or neither'
}
$scoreCutsActive = $PSBoundParameters.ContainsKey('ScoreLowCut')
if ($scoreCutsActive -and $ScoreLowCut -ge $ScoreHighCut) {
    throw "-ScoreLowCut ($ScoreLowCut) must be below -ScoreHighCut ($ScoreHighCut)"
}

# Spectre refuses to draw its box characters on an OEM code page, and this is the fix it asks for.
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

if (-not (Get-Module -ListAvailable -Name PwshSpectreConsole)) {
    throw 'PwshSpectreConsole is missing: Install-Module PwshSpectreConsole -Scope CurrentUser'
}
Import-Module PwshSpectreConsole -ErrorAction Stop
# A class needs the Spectre.Console types to be loaded when it is parsed, and PowerShell parses this
# script before running it, so the renderable lives in its own file and is pulled in here.
. (Join-Path $PSScriptRoot 'SpectreScriptRenderable.ps1')

if ($MaxTickets -lt 0) {
    throw 'MaxTickets must be >= 0.'
}
if ($BatchSize -lt 1) {
    throw 'BatchSize must be >= 1.'
}
if ($MaxWaitSeconds -lt 1) {
    throw 'MaxWaitSeconds must be >= 1.'
}

$data = Get-Content -LiteralPath $InputPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
$questions = $data.questions
if ($QuestionsPath) {
    $questions = Get-Content -LiteralPath $QuestionsPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
    if ($questions.questions) { $questions = $questions.questions }
}
if (-not $questions) { throw 'No questions: put a "questions" object in the input file or pass -QuestionsPath.' }
$tickets = @(if ($data -is [array]) { $data } else { $data.tickets })
if (-not $tickets) { throw 'No tickets found: expected a "tickets" array or a bare array of tickets.' }
if ($MaxTickets -gt 0) { $tickets = @($tickets | Select-Object -First $MaxTickets) }
$missing = $tickets | Where-Object { $null -eq $_.$TextProperty } | Select-Object -First 1
if ($missing) { throw "A ticket has no '$TextProperty' property; use -TextProperty." }

try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
}
catch {
    throw "laya-serve not reachable at $BaseUrl. Start it with: .\.venv\Scripts\python.exe laya_serve_openvino.py"
}
if (-not $health.status) { throw "$BaseUrl does not answer like laya-serve; use jev-client.ps1 for jev_server.py." }

Write-Host "laya-serve at $BaseUrl, device $($health.device), loaded $(@($health.loaded) -join ', ')"
$chunks = for ($start = 0; $start -lt $tickets.Count; $start += $BatchSize) {
    [pscustomobject]@{
        Index   = [int]($start / $BatchSize)
        Tickets = @($tickets[$start..([math]::Min($start + $BatchSize - 1, $tickets.Count - 1))])
    }
}
$questionsJson = $questions | ConvertTo-Json -Depth 20 -Compress
$textProperty = $TextProperty
$chunkTotal = $chunks.Count
$questionCount = @($questions.PSObject.Properties.Name).Count
Write-Host "$($tickets.Count) state(s), model $Model, batch_size $BatchSize"
Write-Host "$($chunks.Count) request(s) of at most $BatchSize state(s)"

# One row per request, written by the request itself and drawn by this thread alone.
$progress = [hashtable]::Synchronized(@{})
foreach ($chunk in $chunks) {
    $progress[$chunk.Index] = @{ Status = 'pending'; States = $chunk.Tickets.Count; Started = $null; Seconds = 0 }
}

function New-SpectreProgress {
    # A request may be writing its own row while this runs, and enumerating a Hashtable that is
    # being written throws. Copy every row under the table's own lock, then read the copy.
    $snapshot = @{}
    [System.Threading.Monitor]::Enter($progress.SyncRoot)
    try {
        foreach ($key in @($progress.Keys)) { $snapshot[$key] = $progress[$key] }
    }
    finally { [System.Threading.Monitor]::Exit($progress.SyncRoot) }

    $rows = foreach ($chunk in $chunks) {
        $row = $snapshot[$chunk.Index]
        # A row still in flight has no duration of its own yet: read the clock instead of showing
        # the number written at its last state change, which is what made it look frozen.
        $inFlight = $row.Status -notin 'done', 'failed'
        if ($inFlight -and $row.Started) { $seconds = ([DateTime]::UtcNow - $row.Started).TotalSeconds }
        else { $seconds = [double]$row.Seconds }
        $duration = if ($seconds -ge 0.05) { '{0:N1}s' -f $seconds } else { '' }
        $perTicket = ''
        $answersPerSecond = ''
        if (-not $inFlight -and $seconds) {
            $perTicket = '{0:N0} ms' -f ($seconds * 1000.0 / [Math]::Max(1, $row.States))
            $answersPerSecond = '{0:N1}' -f ($row.States * $questionCount / $seconds)
        }
        [pscustomobject]@{
            Batch                = $chunk.Index + 1
            Status               = $row.Status
            States               = $row.States
            'Accepted to answer' = $duration
            'Per ticket'         = $perTicket
            'Answers/s'          = $answersPerSecond
        }
    }

    $doneRows = @($snapshot.Values | Where-Object { $_.Status -eq 'done' })
    $done = @($doneRows | ForEach-Object { $_.Seconds })
    $summary = @(('received {0}/{1} batch(es), elapsed {2:N1}s' -f `
            $done.Count, $chunkTotal, $watch.Elapsed.TotalSeconds))
    if ($done.Count) {
        $sorted = @($done | Sort-Object)
        if ($sorted.Count % 2) { $median = $sorted[[int](($sorted.Count - 1) / 2)] }
        else { $median = ($sorted[$sorted.Count / 2 - 1] + $sorted[$sorted.Count / 2]) / 2 }
        $summary += ('min {0:N1}s  median {1:N1}s  mean {2:N1}s  max {3:N1}s' -f `
                $sorted[0], $median, ($done | Measure-Object -Average).Average, $sorted[-1])
    }
    else { $summary += 'min -  median -  mean -  max -' }
    $doneStates = [int](($doneRows | Measure-Object -Property States -Sum).Sum)
    $elapsed = [Math]::Max($watch.Elapsed.TotalSeconds, 0.001)
    $summary += ('throughput {0:N1} state(s)/s, {1:N1} answer(s)/s over {2} state(s) in {3:N1}s' -f `
            ($doneStates / $elapsed), ($doneStates * $questionCount / $elapsed), $doneStates, $elapsed)

    $table = Format-SpectreTable -Data @($rows) `
        -Property 'Batch', 'Status', 'States', 'Accepted to answer', 'Per ticket', 'Answers/s' `
        -Border Rounded -Expand
    # No -Expand on the panel or on Rows: the live region is as tall as the window, and an expanded
    # panel fills all of it, which is what padded the three summary lines out to twenty.
    $panel = Format-SpectrePanel -Data ($summary -join "`n") -Header '[grey]summary[/]'
    return Format-SpectreRows -Data @($table, $panel)
}

$watch = [System.Diagnostics.Stopwatch]::StartNew()
$jobs = foreach ($chunk in $chunks) {
    Start-ThreadJob -ArgumentList $chunk, $questionsJson, $textProperty, $progress, $Model, $BaseUrl, $BatchSize, $TimeoutSeconds, $MaxWaitSeconds -ScriptBlock {
        param($chunk, $questionsJson, $textProperty, $progress, $Model, $BaseUrl, $BatchSize, $TimeoutSeconds, $MaxWaitSeconds)
        $ErrorActionPreference = 'Stop'
        $states = @($chunk.Tickets | ForEach-Object { $_.$textProperty })
        $body = @{
            model      = $Model
            states     = $states
            questions  = ($questionsJson | ConvertFrom-Json -Depth 20)
            batch_size = $BatchSize
        } | ConvertTo-Json -Depth 20 -Compress
        $refusals = 0
        $answers = $null
        $inferenceMs = 0
        $errorMessage = $null
        $waited = 0
        $requestSeconds = 0
        while ($true) {
            # Only the call the server accepted is timed: a refusal is not part of this request's
            # duration, so a row can sit in 'waiting' for a while and still report a short figure.
            $progress[$chunk.Index] = @{ Status = 'sent'; States = $chunk.Tickets.Count; Started = [DateTime]::UtcNow; Seconds = 0 }
            $attempt = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                $response = Invoke-WebRequest -Method Post -Uri "$BaseUrl/v1/systemone/batch" `
                    -ContentType 'application/json; charset=utf-8' -TimeoutSec $TimeoutSeconds `
                    -Body ([Text.Encoding]::UTF8.GetBytes($body))
                $payload = $response.Content | ConvertFrom-Json -Depth 20
                if (@($payload.results).Count -ne $chunk.Tickets.Count) {
                    throw "the server answered $(@($payload.results).Count) result(s) for $($chunk.Tickets.Count) state(s)"
                }
                $inferenceMs = [double]($response.Headers['X-Inference-Time-Ms'] | Select-Object -First 1)
                $answers = @($payload.results | ForEach-Object { $_.answers })
                $errorMessage = $null
                $attempt.Stop()
                $requestSeconds = $attempt.Elapsed.TotalSeconds
                break
            }
            catch {
                $errorMessage = "$($_.Exception.Message) $($_.ErrorDetails.Message)"
                $status = 0
                $wait = 1
                if ($_.Exception.Response) {
                    $status = [int]$_.Exception.Response.StatusCode
                    $retryAfter = $_.Exception.Response.Headers.RetryAfter
                    if ($retryAfter -and $retryAfter.Delta) {
                        $wait = [int][math]::Ceiling($retryAfter.Delta.TotalSeconds)
                    }
                }
                # 503 is the server refusing to admit the request, not a failure: a pass takes
                # seconds, so several refusals in a row are normal. It says how long to wait, and
                # only the time spent waiting is bounded here.
                if ($status -ne 503 -or $waited -ge $MaxWaitSeconds) { break }
                $refusals++
                $waited += $wait
                $progress[$chunk.Index] = @{ Status = 'waiting'; States = $chunk.Tickets.Count; Started = $null; Seconds = 0 }
                Start-Sleep -Seconds $wait
            }
        }
        $seconds = $requestSeconds
        if ($errorMessage -and $refusals -gt 0 -and $waited -ge $MaxWaitSeconds) {
            $errorMessage = "no slot in $waited s: $errorMessage"
        }
        $state = if ($errorMessage) { 'failed' } else { 'done' }
        $progress[$chunk.Index] = @{ Status = $state; States = $chunk.Tickets.Count; Started = $null; Seconds = $seconds }
        [pscustomobject]@{
            Index       = $chunk.Index
            Tickets     = $chunk.Tickets
            Answers     = $answers
            InferenceMs = $inferenceMs
            Seconds     = $seconds
            Refusals    = $refusals
            Error       = $errorMessage
        }
    }
}

# The library holds the object it is handed, so this is a renderable that rebuilds itself on every
# draw: the region follows the data, and it is only as tall as its content.
$live = [SpectreScriptRenderable]::new({ New-SpectreProgress })
Invoke-SpectreLive -Data $live -ScriptBlock {
    param (
        [Spectre.Console.LiveDisplayContext] $Context
    )
    while (@($jobs | Where-Object { $_.State -in 'Running', 'NotStarted' }).Count) {
        $Context.refresh()
        Start-Sleep -Milliseconds $RefreshMs
    }
    $Context.refresh()
}
$watch.Stop()

$outcomes = @()
foreach ($job in $jobs) {
    try { $outcomes += @(Receive-Job -Job $job -ErrorAction Stop) } catch { }
    Remove-Job -Job $job -Force
}
$byIndex = @{}
foreach ($outcome in $outcomes) { $byIndex[[int]$outcome.Index] = $outcome }
$outcomes = foreach ($chunk in $chunks) {
    if ($byIndex.ContainsKey($chunk.Index)) { $byIndex[$chunk.Index] }
    else {
        [pscustomobject]@{
            Index       = $chunk.Index
            Tickets     = $chunk.Tickets
            Answers     = $null
            InferenceMs = 0
            Seconds     = 0
            Refusals    = 0
            Error       = 'the request never came back'
        }
    }
}
$refusals = [int](($outcomes | Measure-Object -Property Refusals -Sum).Sum)

# Chunks come back in the order the server admitted them, so the ticket order is restored here.
$results = @()
foreach ($outcome in @($outcomes | Sort-Object -Property Index)) {
    for ($i = 0; $i -lt $outcome.Tickets.Count; $i++) {
        $answer = $null
        if (-not $outcome.Error) { $answer = $outcome.Answers[$i] }
        $results += [pscustomobject]@{
            Ticket      = $outcome.Tickets[$i]
            Answers     = $answer
            InferenceMs = [double]$outcome.InferenceMs
            Error       = $outcome.Error
        }
    }
}

function Get-ScoreLevel($Answer) {
    # -ScoreLowCut and -ScoreHighCut place the two boundaries on the score number: at or below the
    # first is the lowest level, above the second the highest, in between the middle one (3-level
    # questions). Without them the level is the one the model favours, as this client always did.
    if ($scoreCutsActive) {
        $value = [double]$Answer.score
        if ($value -le $ScoreLowCut) { return 0 }
        if ($value -gt $ScoreHighCut) { return 2 }
        return 1
    }
    $level = 0
    $best = -1
    foreach ($key in $Answer.probabilities.PSObject.Properties.Name) {
        $p = [double]$Answer.probabilities.$key
        if ($p -gt $best) { $best = $p; $level = [int]$key }
    }
    return $level
}

function Test-Answer($Definition, $Answer, $Expected) {
    switch ($Definition.type) {
        'choice' { return $Answer.choice -eq $Expected }
        'noul' { return ([double]$Answer.noul -ge $NoulThreshold) -eq [bool]$Expected }
        'score' {
            # The average of levels weighted by probability can round to a level neither close
            # to being this model's top choice; read the level it actually favors instead.
            $level = Get-ScoreLevel $Answer
            $label = ($Definition.criteria[$level] -split ':')[0].Trim()
            return $label -eq $Expected
        }
    }
}

function Get-ExpectedProbability($Definition, $Answer, $Expected) {
    # How much probability the model put on the right answer, whether or not it won: a
    # 3-level score question is pure guessing at exactly 1/3, so this is what separates
    # "no signal" from "signal lost to the decision rule".
    switch ($Definition.type) {
        'choice' { return [double]$Answer.probabilities.$Expected }
        'noul' {
            $pTrue = [double]$Answer.noul
            if ([bool]$Expected) { return $pTrue }
            return 1.0 - $pTrue
        }
        'score' {
            $index = $null
            for ($i = 0; $i -lt $Definition.criteria.Count; $i++) {
                $label = ($Definition.criteria[$i] -split ':')[0].Trim()
                if ($label -eq $Expected) { $index = $i; break }
            }
            if ($null -eq $index) { return $null }
            return [double]$Answer.probabilities."$index"
        }
    }
}

function Get-ChosenLabel($Definition, $Answer) {
    # What Laya actually picked, independent of whether it was right: used to tell a model that
    # varies its guess (but uncorrelated with truth) apart from one that just repeats one answer.
    switch ($Definition.type) {
        'choice' { return $Answer.choice }
        'noul' {
            if ([double]$Answer.noul -ge $NoulThreshold) { return 'true' }
            return 'false'
        }
        'score' {
            $level = Get-ScoreLevel $Answer
            return ($Definition.criteria[$level] -split ':')[0].Trim()
        }
    }
}

$failed = @($results | Where-Object { $_.Error })
$answered = @($results | Where-Object { -not $_.Error })
$scored = @($answered | Where-Object { $_.Ticket.expected })
$accuracy = foreach ($name in $questions.PSObject.Properties.Name) {
    $correct = @($scored | Where-Object { Test-Answer $questions.$name $_.Answers.$name $_.Ticket.expected.$name }).Count
    $expectedProbs = @($scored | ForEach-Object { Get-ExpectedProbability $questions.$name $_.Answers.$name $_.Ticket.expected.$name } |
            Where-Object { $null -ne $_ })
    $meanExpectedProb = if ($expectedProbs.Count) { ($expectedProbs | Measure-Object -Average).Average } else { $null }
    # answer_confidence (max probability) is the calibrated quantity Laya's own docs say to use
    # for this; confidence (entropy-based) is explicitly documented as not comparable to it.
    $confidences = @($scored | ForEach-Object { $_.Answers.$name.answer_confidence } | Where-Object { $null -ne $_ })
    $meanConfidence = if ($confidences.Count) { ($confidences | Measure-Object -Average).Average } else { $null }
    $chosen = @($scored | ForEach-Object { Get-ChosenLabel $questions.$name $_.Answers.$name })
    $distribution = ($chosen | Group-Object | Sort-Object Name | ForEach-Object { "$($_.Name):$($_.Count)" }) -join ' '
    # What the dataset itself looks like: a 333/333/333 split of Laya's picks only means
    # "uncorrelated with truth" if the real labels are themselves close to that split.
    $expectedLabels = @($scored | ForEach-Object { $_.Ticket.expected.$name })
    $expectedDistribution = ($expectedLabels | Group-Object | Sort-Object Name | ForEach-Object { "$($_.Name):$($_.Count)" }) -join ' '
    [pscustomobject]@{
        Question             = $name
        Type                 = $questions.$name.type
        Correct              = $correct
        Scored               = $scored.Count
        Accuracy             = if ($scored.Count) { '{0:P1}' -f ($correct / $scored.Count) } else { 'n/a' }
        'MeanP(expected)'    = if ($null -ne $meanExpectedProb) { '{0:P1}' -f $meanExpectedProb } else { 'n/a' }
        MeanConfidence       = if ($null -ne $meanConfidence) { '{0:P1}' -f $meanConfidence } else { 'n/a' }
        LayaDistribution     = $distribution
        ExpectedDistribution = $expectedDistribution
    }
}

$failedRequests = @($outcomes | Where-Object { $_.Error })
$inference = @($outcomes | Where-Object { -not $_.Error } | ForEach-Object { $_.InferenceMs })
Write-Host ''
Write-Host ("{0} request(s) sent, {1} refused, {2} state(s) in {3:N1}s ({4:N1} state(s)/s, {5:N1} answer(s)/s), {6} failed request(s), {7} failed state(s)" -f `
        $chunks.Count, $refusals, $tickets.Count, $watch.Elapsed.TotalSeconds,
        ($tickets.Count / [math]::Max($watch.Elapsed.TotalSeconds, 0.001)),
        ($tickets.Count * $questionCount / [math]::Max($watch.Elapsed.TotalSeconds, 0.001)),
        $failedRequests.Count, $failed.Count)
if ($inference.Count) {
    Write-Host ("answers: {0} = {1} state(s) x {2} question(s)" -f `
            ($tickets.Count * $questionCount), $tickets.Count, $questionCount)
    Write-Host ("server-reported inference: {0} of {1} request(s) answered, sum {2:N0} ms" -f `
            $inference.Count, $chunks.Count, ($inference | Measure-Object -Sum).Sum)
}
if ($failed.Count -lt $tickets.Count) {
    Write-Host ''
    Write-Host "routed to $($health.loaded -join ', ')"
}
# `Format-Table -AutoSize` silently drops the columns that do not fit the console, which is how
# ExpectedDistribution disappeared and LayaDistribution came back cut with an ellipsis. Spectre
# wraps a cell instead of dropping it, so nothing is lost whatever the window width. Two tables
# rather than one: the two distribution columns need about seventy characters between them, and in
# a single nine-column table they take that room from the numbers, squeezing Question and the
# header row down to one character each. Rendered with the same renderer as the live table above.
$accuracyNumbers = Format-SpectreTable -Data @($accuracy) `
    -Property 'Question', 'Type', 'Correct', 'Scored', 'Accuracy', 'MeanP(expected)', 'MeanConfidence' `
    -Border Rounded -Expand -Wrap
$accuracyDistributions = Format-SpectreTable -Data @($accuracy) `
    -Property 'Question', 'ExpectedDistribution', 'LayaDistribution' `
    -Border Rounded -Expand -Wrap
Format-SpectreRows -Data @($accuracyNumbers, $accuracyDistributions)
$failed | Select-Object -First 3 | ForEach-Object { Write-Warning "Ticket $($_.Ticket.$IdProperty): $($_.Error)" }
if ($OutputPath) {
    $results | ForEach-Object {
        $ticket = $_.Ticket
        $matches = $null
        if (-not $_.Error -and $ticket.expected) {
            $matches = [ordered]@{}
            foreach ($name in $questions.PSObject.Properties.Name) {
                $matches[$name] = [bool](Test-Answer $questions.$name $_.Answers.$name $ticket.expected.$name)
            }
        }
        [pscustomobject]@{
            id       = $ticket.$IdProperty
            answers  = $_.Answers
            expected = $ticket.expected
            match    = $matches
            error    = $_.Error
        }
    } | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
    Write-Host "Results written to $OutputPath"
}
