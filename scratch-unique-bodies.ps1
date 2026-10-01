$data = Get-Content -LiteralPath '.\tickets-1000.json' -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
$bodies = foreach ($t in $data.tickets) {
    $idx = $t.state.IndexOf('TKT-')
    $before = if ($idx -ge 0) { $t.state.Substring(0, $idx) } else { $t.state }
    $sentences = $before -split '(?<=[.?!])\s+' | Where-Object { $_.Trim().Length -gt 0 }
    if ($sentences.Count -ge 3) { ($sentences[1..($sentences.Count - 2)] -join ' ').Trim() }
}
$unique = $bodies | Sort-Object -Unique
Write-Output ("total tickets: {0}" -f $data.tickets.Count)
Write-Output ("unique problem-description sentences to translate: {0}" -f $unique.Count)
$unique | Select-Object -First 8
