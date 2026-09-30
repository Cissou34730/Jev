#Requires -Version 7.0
<#
.SYNOPSIS
Drives laya-serve (laya_serve_openvino.py) and scores the answers against 'expected' when present.

.DESCRIPTION
Written for laya-serve (laya_serve_openvino.py) and nothing else. The whole file goes in ONE request
to POST /v1/systemone/batch, which carries every state at once: {model, states: [...], questions}
answered by {model, results: [...], usage}. Grouping the states into forward passes is the server's
business, so there is no concurrency to set here.

The input is either {"questions": {...}, "tickets": [...]} or a bare array of tickets (then use
-QuestionsPath). -Model defaults to "multilingual", a name laya-serve routes; leave it alone
unless you know which checkpoint name you want.

.EXAMPLE
.\laya-client.ps1 -InputPath .\tickets.json
.\laya-client.ps1 -InputPath .\tickets.json -MaxTickets 20
.\laya-client.ps1 -InputPath .\tickets.json -QuestionsPath .\questions.json -TextProperty description
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
    [int]$TimeoutSeconds = 3600,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'

if ($MaxTickets -lt 0) {
    throw 'MaxTickets must be >= 0.'
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
Write-Host "$($tickets.Count) state(s), model $Model, one request"
$states = @($tickets | ForEach-Object { $_.$TextProperty })
$body = @{ model = $Model; states = $states; questions = $questions } | ConvertTo-Json -Depth 20 -Compress
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$inferenceMs = 0
$results = @()
try {
    $response = Invoke-WebRequest -Method Post -Uri "$BaseUrl/v1/systemone/batch" `
        -ContentType 'application/json; charset=utf-8' -TimeoutSec $TimeoutSeconds `
        -Body ([Text.Encoding]::UTF8.GetBytes($body))
    $payload = $response.Content | ConvertFrom-Json -Depth 20
    $inferenceMs = [double]($response.Headers['X-Inference-Time-Ms'] | Select-Object -First 1)
    if (@($payload.results).Count -ne $tickets.Count) {
        throw "the server answered $(@($payload.results).Count) result(s) for $($tickets.Count) state(s)"
    }
    for ($i = 0; $i -lt $tickets.Count; $i++) {
        $results += [pscustomobject]@{
            Ticket      = $tickets[$i]
            Answers     = $payload.results[$i].answers
            InferenceMs = $inferenceMs
            Error       = $null
        }
    }
}
catch {
    $message = "$($_.Exception.Message) $($_.ErrorDetails.Message)"
    foreach ($ticket in $tickets) {
        $results += [pscustomobject]@{
            Ticket      = $ticket
            Answers     = $null
            InferenceMs = 0
            Error       = $message
        }
    }
}
$watch.Stop()

function Test-Answer($Definition, $Answer, $Expected) {
    switch ($Definition.type) {
        'choice' { return $Answer.choice -eq $Expected }
        'noul' { return ([double]$Answer.noul -ge 0.5) -eq [bool]$Expected }
        'score' {
            $level = [int][math]::Round([double]$Answer.score, [MidpointRounding]::AwayFromZero)
            return $Definition.criteria[$level] -eq $Expected
        }
    }
}

$failed = @($results | Where-Object { $_.Error })
$answered = @($results | Where-Object { -not $_.Error })
$scored = @($answered | Where-Object { $_.Ticket.expected })
$accuracy = foreach ($name in $questions.PSObject.Properties.Name) {
    $correct = @($scored | Where-Object { Test-Answer $questions.$name $_.Answers.$name $_.Ticket.expected.$name }).Count
    [pscustomobject]@{
        Question = $name
        Type     = $questions.$name.type
        Correct  = $correct
        Scored   = $scored.Count
        Accuracy = if ($scored.Count) { '{0:P1}' -f ($correct / $scored.Count) } else { 'n/a' }
    }
}

$inference = @($answered | ForEach-Object { $_.InferenceMs })
Write-Host ''
Write-Host ("1 request, {0} state(s) in {1:N1}s ({2:N1} states/s), {3} failed" -f `
        $tickets.Count, $watch.Elapsed.TotalSeconds,
        ($tickets.Count / [math]::Max($watch.Elapsed.TotalSeconds, 0.001)), $failed.Count)
if ($inference.Count) {
    Write-Host ("items: {0} = {1} state(s) x {2} question(s)" -f `
            $inference.Count, $tickets.Count, $questions.PSObject.Properties.Name.Count)
    Write-Host ("server-reported inference for the whole request: {0:N0} ms (first one pays the model compile)" -f $inference[0])
}
if ($failed.Count -lt $tickets.Count) {
    Write-Host ''
    Write-Host "routed to $($health.loaded -join ', ')"
}
$accuracy | Format-Table -AutoSize
$failed | Select-Object -First 3 | ForEach-Object { Write-Warning "Ticket $($_.Ticket.$IdProperty): $($_.Error)" }

if ($OutputPath) {
    $results | ForEach-Object { [pscustomobject]@{ id = $_.Ticket.$IdProperty; answers = $_.Answers; error = $_.Error } } |
        ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
    Write-Host "Results written to $OutputPath"
}
