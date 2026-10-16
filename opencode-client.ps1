#Requires -Version 7.0
<#
.SYNOPSIS
Scores tickets against the Jev System-One models served by the opencode endpoint, with the live
Spectre display and prediction measures of clef-client.ps1.

.DESCRIPTION
The input is either {"questions": {...}, "tickets": [...]} or a bare array of tickets (then use
-QuestionsPath). The display shows live requests, throughput, per-question accuracy, probability of
the expected answer, confidence, and expected/predicted distributions.

opencode serves TypeSafe's Jev models as classifiers (api "typesafe-system-one"): jev-1.13 and
jev-1.13-free. The route is POST {ApiBase}/systemone with the Jev body {model, state, questions},
answered with {model, answers, usage}. One state per request, so each ticket is sent separately
with bounded concurrency.

The key comes from -ApiKey, else OPENCODE_API_KEY, else the "opencode" entry of -AuthPath
(~/.pi/agent/auth.json by default). It is never written to disk.

opencode publishes no rate limit, and a 60-request burst all answered 200, so no throttle is
assumed here. HTTP 429 and 503 are treated as "slow down": the request is retried after the
Retry-After delay, up to -MaxWaitSeconds in total per ticket.

Requires PwshSpectreConsole and SpectreScriptRenderable.ps1, as laya-client-spectre.ps1 does.

.EXAMPLE
.\opencode-client.ps1 -Limit 20
.\opencode-client.ps1 -InputPath .\tickets-1000.json -Concurrency 8 -OutputPath resultats-opencode-1000.json
.\opencode-client.ps1 -InputPath .\tickets-array.json -QuestionsPath .\tickets-1000.json -MaxTickets 5
#>
[CmdletBinding()]
param(
    [Alias('TicketsFile')]
    [string]$InputPath = (Join-Path $PSScriptRoot 'tickets-1000.json'),
    [string]$QuestionsPath,
    [string]$TextProperty = 'state',
    [string]$IdProperty = 'id',
    [string]$ApiBase = 'https://opencode.ai/inference/v1',
    [string]$ServerUrl,
    [Alias('Limit')]
    [int]$MaxTickets = 0,
    [int]$Concurrency = 4,
    [string]$Model = 'jev-1.13',
    [string]$ApiKey,
    [string]$AuthPath = (Join-Path $HOME '.pi/agent/auth.json'),
    [int]$MaxWaitSeconds = 300,
    [int]$RefreshMs = 200,
    [int]$TimeoutSeconds = 3600,
    [Alias('OutFile')]
    [string]$OutputPath,
    [double]$NoulThreshold = 0.5,
    [double]$InputCostPerMillion = 0.042,
    [double]$OutputCostPerMillion = 0,
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
if ($MaxTickets -lt 0 -or $Concurrency -lt 1 -or $MaxWaitSeconds -lt 1 -or
    $RefreshMs -lt 1 -or $TimeoutSeconds -lt 1 -or $NoulThreshold -lt 0 -or $NoulThreshold -gt 1) {
    throw 'MaxTickets must be >= 0, Concurrency/MaxWaitSeconds/RefreshMs/TimeoutSeconds >= 1, NoulThreshold between 0 and 1.'
}
if ($PSBoundParameters.ContainsKey('ServerUrl')) {
    if ($PSBoundParameters.ContainsKey('ApiBase')) {
        throw 'Pass either -ServerUrl or -ApiBase, not both.'
    }
    if ($ServerUrl -notmatch '^https?://.+/systemone/?$') {
        throw '-ServerUrl must be the full /systemone endpoint URL.'
    }
    $ApiBase = $ServerUrl -replace '/systemone/?$', ''
}

# -ApiKey, else the environment, else the opencode entry of the pi auth store.
if (-not $ApiKey) { $ApiKey = $env:OPENCODE_API_KEY }
if (-not $ApiKey) {
    if (-not (Test-Path -LiteralPath $AuthPath)) {
        throw "No API key: pass -ApiKey, set OPENCODE_API_KEY, or provide -AuthPath (looked for $AuthPath)."
    }
    $auth = Get-Content -LiteralPath $AuthPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
    $entry = $auth.PSObject.Properties['opencode']
    if (-not $entry) { throw "$AuthPath has no 'opencode' entry." }
    $ApiKey = $entry.Value.key
    if (-not $ApiKey) { throw "The 'opencode' entry of $AuthPath has no key." }
}

$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
if (-not (Get-Module -ListAvailable -Name PwshSpectreConsole)) {
    throw 'PwshSpectreConsole is missing: Install-Module PwshSpectreConsole -Scope CurrentUser'
}
Import-Module PwshSpectreConsole -ErrorAction Stop
. (Join-Path $PSScriptRoot 'SpectreScriptRenderable.ps1')

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

Write-Host "opencode Jev at $ApiBase, model $Model"
$questionCount = @($questions.PSObject.Properties.Name).Count
Write-Host "$($tickets.Count) state(s), $($tickets.Count) request(s), concurrency $Concurrency"
$progress = [hashtable]::Synchronized(@{})
$indexedTickets = for ($i = 0; $i -lt $tickets.Count; $i++) {
    $progress[$i] = @{ Status = 'pending'; Started = $null; Seconds = 0 }
    [pscustomobject]@{ Index = $i; Ticket = $tickets[$i] }
}
$questionsJson = $questions | ConvertTo-Json -Depth 20 -Compress

function New-SpectreProgress {
    $snapshot = @{}
    [System.Threading.Monitor]::Enter($progress.SyncRoot)
    try {
        foreach ($key in @($progress.Keys)) { $snapshot[$key] = $progress[$key] }
    }
    finally { [System.Threading.Monitor]::Exit($progress.SyncRoot) }

    $doneRows = @($snapshot.Values | Where-Object { $_.Status -eq 'done' })
    $done = @($doneRows | ForEach-Object { $_.Seconds })
    $completed = @($snapshot.Keys | Where-Object { $snapshot[$_].Status -in 'done', 'failed' } | Sort-Object)
    $active = @($snapshot.Keys | Where-Object { $snapshot[$_].Status -in 'sent', 'waiting' } | Sort-Object)
    $pending = @($snapshot.Keys | Where-Object { $snapshot[$_].Status -eq 'pending' } | Sort-Object | Select-Object -First 3)
    $visible = @($completed | Select-Object -Last 12) + $active + $pending | Sort-Object -Unique
    $rows = foreach ($index in $visible) {
        $row = $snapshot[$index]
        $seconds = if ($row.Started) { ([DateTime]::UtcNow - $row.Started).TotalSeconds }
            else { [double]$row.Seconds }
        $duration = if ($seconds -ge 0.05) { '{0:N1}s' -f $seconds } else { '' }
        $perTicket = ''
        $answersPerSecond = ''
        if ($row.Status -eq 'done' -and $seconds) {
            $perTicket = '{0:N0} ms' -f ($seconds * 1000.0)
            $answersPerSecond = '{0:N1}' -f ($questionCount / $seconds)
        }
        [pscustomobject]@{
            Request              = $index + 1
            Status               = $row.Status
            States               = 1
            'Accepted to answer' = $duration
            'Per ticket'         = $perTicket
            'Answers/s'          = $answersPerSecond
        }
    }

    $summary = @(('received {0}/{1} request(s), elapsed {2:N1}s' -f `
            $completed.Count, $tickets.Count, $watch.Elapsed.TotalSeconds))
    if ($done.Count) {
        $sorted = @($done | Sort-Object)
        if ($sorted.Count % 2) { $median = $sorted[[int](($sorted.Count - 1) / 2)] }
        else { $median = ($sorted[$sorted.Count / 2 - 1] + $sorted[$sorted.Count / 2]) / 2 }
        $summary += ('min {0:N1}s  median {1:N1}s  mean {2:N1}s  max {3:N1}s' -f `
                $sorted[0], $median, ($done | Measure-Object -Average).Average, $sorted[-1])
    }
    else { $summary += 'min -  median -  mean -  max -' }
    $elapsed = [Math]::Max($watch.Elapsed.TotalSeconds, 0.001)
    $summary += ('throughput {0:N1} state(s)/s, {1:N1} answer(s)/s over {2} state(s) in {3:N1}s' -f `
            ($done.Count / $elapsed), ($done.Count * $questionCount / $elapsed), $done.Count, $elapsed)

    $table = Format-SpectreTable -Data @($rows) `
        -Property 'Request', 'Status', 'States', 'Accepted to answer', 'Per ticket', 'Answers/s' `
        -Border Rounded -Expand
    $panel = Format-SpectrePanel -Data ($summary -join "`n") -Header '[grey]summary[/]'
    return Format-SpectreRows -Data @($table, $panel)
}

$watch = [System.Diagnostics.Stopwatch]::StartNew()
$job = $indexedTickets | ForEach-Object -ThrottleLimit $Concurrency -Parallel {
    $item = $_
    $text = $using:TextProperty
    $questionSet = $using:questionsJson | ConvertFrom-Json -Depth 20
    $questionNames = @($questionSet.PSObject.Properties.Name)
    $uri = $using:ApiBase
    $timeout = $using:TimeoutSeconds
    $modelName = $using:Model
    $key = $using:ApiKey
    $progressRows = $using:progress
    $maxWait = $using:MaxWaitSeconds
    $body = @{ model = $modelName; state = $item.Ticket.$text; questions = $questionSet } |
        ConvertTo-Json -Depth 20 -Compress
    $headers = @{ Authorization = "Bearer $key" }
    $refusals = 0
    $waited = 0
    $errorMessage = $null
    $answers = $null
    $usage = $null
    $requestSeconds = 0
    while ($true) {
        $progressRows[$item.Index] = @{ Status = 'sent'; Started = [DateTime]::UtcNow; Seconds = 0 }
        $attempt = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $response = Invoke-WebRequest -Method Post -Uri "$uri/systemone" `
                -Headers $headers -ContentType 'application/json; charset=utf-8' -TimeoutSec $timeout `
                -Body ([Text.Encoding]::UTF8.GetBytes($body))
            $payload = $response.Content | ConvertFrom-Json -Depth 20
            if (-not $payload.answers) { throw 'the endpoint returned no answers' }
            foreach ($name in $questionNames) {
                if ($null -eq $payload.answers.PSObject.Properties[$name]) {
                    throw "the endpoint returned no answer for '$name'"
                }
                $answer = $payload.answers.$name
                if ($answer.type -ne $questionSet.$name.type) {
                    throw "the endpoint returned the wrong answer type for '$name'"
                }
                if ($answer.type -in 'choice', 'score' -and
                    (-not $answer.probabilities -or -not @($answer.probabilities.PSObject.Properties).Count)) {
                    throw "the endpoint returned no probabilities for '$name'"
                }
                if ($answer.type -eq 'noul' -and $null -eq $answer.PSObject.Properties['noul']) {
                    throw "the endpoint returned no noul value for '$name'"
                }
            }
            $answers = $payload.answers
            $usage = $payload.usage
            $attempt.Stop()
            $requestSeconds = $attempt.Elapsed.TotalSeconds
            $errorMessage = $null
            break
        }
        catch {
            $errorMessage = "$($_.Exception.Message) $($_.ErrorDetails.Message)"
            $answers = $null
            $usage = $null
            $status = 0
            $wait = 1
            if ($_.Exception.Response) {
                $status = [int]$_.Exception.Response.StatusCode
                $retryAfter = $_.Exception.Response.Headers.RetryAfter
                if ($retryAfter -and $retryAfter.Delta) {
                    $wait = [int][math]::Ceiling($retryAfter.Delta.TotalSeconds)
                }
            }
            # Only "slow down" is worth waiting for; anything else is this ticket's own failure.
            if ($status -notin 429, 503 -or $waited + $wait -gt $maxWait) { break }
            $refusals++
            $waited += $wait
            $progressRows[$item.Index] = @{ Status = 'waiting'; Started = $null; Seconds = 0 }
            Start-Sleep -Seconds $wait
        }
    }
    if ($errorMessage -and $refusals -gt 0 -and $waited -ge $maxWait) {
        $errorMessage = "no slot in $waited s: $errorMessage"
    }
    $progressRows[$item.Index] = @{
        Status = if ($errorMessage) { 'failed' } else { 'done' }
        Started = $null
        Seconds = $requestSeconds
    }
    [pscustomobject]@{
        Index = $item.Index; Ticket = $item.Ticket; Answers = $answers; Error = $errorMessage
        Usage = $usage; Seconds = $requestSeconds; Refusals = $refusals
    }
} -AsJob

$live = [SpectreScriptRenderable]::new({ New-SpectreProgress })
Invoke-SpectreLive -Data $live -ScriptBlock {
    param([Spectre.Console.LiveDisplayContext] $Context)
    while ($job.State -in 'Running', 'NotStarted') {
        $Context.refresh()
        Start-Sleep -Milliseconds $RefreshMs
    }
    $Context.refresh()
}
$watch.Stop()
$results = @(Receive-Job -Job $job -ErrorAction Stop | Sort-Object -Property Index)
Remove-Job -Job $job -Force
if ($results.Count -ne $tickets.Count) {
    throw "the client collected $($results.Count) result(s) for $($tickets.Count) ticket(s)"
}

function Get-ScoreLevel($Answer) {
    if ($scoreCutsActive) {
        $value = [double]$Answer.score
        if ($value -le $ScoreLowCut) { return 0 }
        if ($value -gt $ScoreHighCut) { return 2 }
        return 1
    }
    $level = 0
    $best = -1
    foreach ($keyOuter in $Answer.probabilities.PSObject.Properties.Name) {
        $p = [double]$Answer.probabilities.$keyOuter
        if ($p -gt $best) { $best = $p; $level = [int]$keyOuter }
    }
    return $level
}

function Test-Answer($Definition, $Answer, $Expected) {
    switch ($Definition.type) {
        'choice' { return $Answer.choice -eq $Expected }
        'noul' { return ([double]$Answer.noul -ge $NoulThreshold) -eq [bool]$Expected }
        'score' {
            $level = Get-ScoreLevel $Answer
            $label = ($Definition.criteria[$level] -split ':')[0].Trim()
            return $label -eq $Expected
        }
    }
}

function Get-ExpectedProbability($Definition, $Answer, $Expected) {
    switch ($Definition.type) {
        'choice' {
            if ($null -eq $Answer.probabilities.PSObject.Properties[$Expected]) {
                throw "unknown expected choice label: $Expected"
            }
            return [double]$Answer.probabilities.$Expected
        }
        'noul' {
            $pTrue = [double]$Answer.noul
            if ([bool]$Expected) { return $pTrue }
            return 1.0 - $pTrue
        }
        'score' {
            for ($i = 0; $i -lt $Definition.criteria.Count; $i++) {
                $label = ($Definition.criteria[$i] -split ':')[0].Trim()
                if ($label -eq $Expected) { return [double]$Answer.probabilities."$i" }
            }
            throw "unknown expected score label: $Expected"
        }
    }
}

function Get-AnswerConfidence($Definition, $Answer) {
    # Jev answers carry a confidence for choice and score; for noul the winning side is derived.
    if ($Definition.type -eq 'noul') {
        $p = [double]$Answer.noul
        return [math]::Max($p, 1.0 - $p)
    }
    if ($null -ne $Answer.PSObject.Properties['confidence']) { return [double]$Answer.confidence }
    return (@($Answer.probabilities.PSObject.Properties.Value | ForEach-Object { [double]$_ }) |
        Measure-Object -Maximum).Maximum
}

function Get-ChosenLabel($Definition, $Answer) {
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
    $expectedProbs = @($scored | ForEach-Object { Get-ExpectedProbability $questions.$name $_.Answers.$name $_.Ticket.expected.$name })
    $meanExpectedProb = if ($expectedProbs.Count) { ($expectedProbs | Measure-Object -Average).Average } else { $null }
    $confidences = @($scored | ForEach-Object { Get-AnswerConfidence $questions.$name $_.Answers.$name })
    $meanConfidence = if ($confidences.Count) { ($confidences | Measure-Object -Average).Average } else { $null }
    $chosen = @($scored | ForEach-Object { Get-ChosenLabel $questions.$name $_.Answers.$name })
    $distribution = ($chosen | Group-Object | Sort-Object Name | ForEach-Object { "$($_.Name):$($_.Count)" }) -join ' '
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
        ModelDistribution    = $distribution
        ExpectedDistribution = $expectedDistribution
    }
}

$refusals = [int](($results | Measure-Object -Property Refusals -Sum).Sum)
$inputTokens = [int](($results | Where-Object { $_.Usage } | ForEach-Object { [int]$_.Usage.input_tokens } | Measure-Object -Sum).Sum)
$outputTokens = [int](($results | Where-Object { $_.Usage } | ForEach-Object { [int]$_.Usage.output_tokens } | Measure-Object -Sum).Sum)
$estimatedCost = $inputTokens / 1e6 * $InputCostPerMillion + $outputTokens / 1e6 * $OutputCostPerMillion
Write-Host ''
Write-Host ("{0} request(s) sent, {1} retried after a rate limit, {2} state(s) in {3:N1}s ({4:N1} state(s)/s, {5:N1} answer(s)/s), {6} failed state(s)" -f `
        $tickets.Count, $refusals, $tickets.Count, $watch.Elapsed.TotalSeconds,
        ($answered.Count / [math]::Max($watch.Elapsed.TotalSeconds, 0.001)),
        ($answered.Count * $questionCount / [math]::Max($watch.Elapsed.TotalSeconds, 0.001)),
        $failed.Count)
if ($answered.Count) {
    Write-Host ("answers: {0} = {1} state(s) x {2} question(s)" -f `
            ($answered.Count * $questionCount), $answered.Count, $questionCount)
}
Write-Host ("usage: {0:N0} input token(s), {1:N0} output token(s), estimated cost {2:C4} at {3}/{4} per million" -f `
        $inputTokens, $outputTokens, $estimatedCost, $InputCostPerMillion, $OutputCostPerMillion)
Write-Host ''
$accuracyNumbers = Format-SpectreTable -Data @($accuracy) `
    -Property 'Question', 'Type', 'Correct', 'Scored', 'Accuracy', 'MeanP(expected)', 'MeanConfidence' `
    -Border Rounded -Expand -Wrap
$accuracyDistributions = Format-SpectreTable -Data @($accuracy) `
    -Property 'Question', 'ExpectedDistribution', 'ModelDistribution' `
    -Border Rounded -Expand -Wrap
Format-SpectreRows -Data @($accuracyNumbers, $accuracyDistributions)
$failed | Select-Object -First 3 | ForEach-Object { Write-Warning "Ticket $($_.Ticket.$IdProperty): $($_.Error)" }

if ($OutputPath) {
    $outputRows = @($results | ForEach-Object {
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
            usage    = $_.Usage
            error    = $_.Error
        }
    })
    ConvertTo-Json -InputObject $outputRows -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
    Write-Host "Results written to $OutputPath"
}
