$data = Get-Content -LiteralPath '.\tickets-1000.json' -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
$rows = foreach ($t in $data.tickets) {
    if (-not $t.state) { continue }
    $idx = $t.state.IndexOf('TKT-')
    $before = if ($idx -ge 0) { $t.state.Substring(0, $idx) } else { $t.state }
    $sentences = $before -split '(?<=[.?!])\s+' | Where-Object { $_.Trim().Length -gt 0 }
    $n = $sentences.Count
    $phrase = if ($n -ge 2) { $sentences[$n - 2].Trim() } else { '<none>' }
    [pscustomobject]@{ Phrase = $phrase; Urgence = $t.expected.urgence }
}
$rows | Group-Object Phrase | Sort-Object Count -Descending | ForEach-Object {
    $labels = ($_.Group | Group-Object Urgence | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ' '
    "{0,4}  {1}" -f $_.Count, $labels
    "      $($_.Name)"
}
