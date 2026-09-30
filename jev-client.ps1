#Requires -Version 7.0
<#
.SYNOPSIS
Sends the tickets of a JSON file to jev_server.py (POST /v1/systemone) and scores the answers against 'expected' when present.

.DESCRIPTION
The input is either {"questions": {...}, "tickets": [...]} or a bare array of tickets (then use -QuestionsPath).
One request per ticket, sent concurrently (the pattern of the official Jev cookbooks): each
request carries one state and the shared questions, and the server collates the rows of the
requests in flight into full forward passes.

.EXAMPLE
.\jev-client.ps1 -InputPath .\tickets.json -MaxTickets 100
.\jev-client.ps1 -InputPath .\tickets.json -QuestionsPath .\questions.json -TextProperty description
.\jev-client.ps1 -InputPath .\tickets.json -Concurrency 50 -MaxTickets 200
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$InputPath,
    [string]$QuestionsPath,
    [string]$TextProperty = 'state',
    [string]$IdProperty = 'id',
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [int]$MaxTickets = 0,
    [int]$Concurrency = 8,
    [string]$Model = 'jev-latest',
    [int]$TimeoutSeconds = 3600,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'

if ($MaxTickets -lt 0 -or $Concurrency -lt 1) {
    throw 'MaxTickets must be >= 0 and Concurrency >= 1.'
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
    Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5 | Out-Null
}
catch {
    throw "Server not reachable at $BaseUrl. Start it with: .\.venv\Scripts\python.exe jev_server.py"
}

Write-Host "$($tickets.Count) ticket(s), concurrency $Concurrency -> $BaseUrl"
$watch = [System.Diagnostics.Stopwatch]::StartNew()

$requestCount = $tickets.Count
$results = $tickets | ForEach-Object -ThrottleLimit $Concurrency -Parallel {
    $ticket = $_
    $text = $using:TextProperty
    $questionSet = $using:questions
    $uri = $using:BaseUrl
    $timeout = $using:TimeoutSeconds
    $modelName = $using:Model
    $body = @{ model = $modelName; state = $ticket.$text; questions = $questionSet } | ConvertTo-Json -Depth 20 -Compress
    try {
        $response = Invoke-RestMethod -Method Post -Uri "$uri/v1/systemone" `
            -ContentType 'application/json; charset=utf-8' -TimeoutSec $timeout `
            -Body ([Text.Encoding]::UTF8.GetBytes($body))
        [pscustomobject]@{ Ticket = $ticket; Answers = $response.answers; Error = $null }
    }
    catch {
        [pscustomobject]@{ Ticket = $ticket; Answers = $null; Error = "$($_.Exception.Message) $($_.ErrorDetails.Message)" }
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
$scored = @($results | Where-Object { -not $_.Error -and $_.Ticket.expected })
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

Write-Host ''
Write-Host ("{0} request(s), {1} ticket(s) in {2:N1}s ({3:N1} tickets/s), {4} failed" -f `
        $requestCount, $tickets.Count, $watch.Elapsed.TotalSeconds, ($tickets.Count / [math]::Max($watch.Elapsed.TotalSeconds, 0.001)), $failed.Count)
$accuracy | Format-Table -AutoSize
$failed | Select-Object -First 3 | ForEach-Object { Write-Warning "Ticket $($_.Ticket.$IdProperty): $($_.Error)" }

if ($OutputPath) {
    $results | ForEach-Object { [pscustomobject]@{ id = $_.Ticket.$IdProperty; answers = $_.Answers; error = $_.Error } } |
        ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
    Write-Host "Results written to $OutputPath"
}
