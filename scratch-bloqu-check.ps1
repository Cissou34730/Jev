$data = Get-Content -LiteralPath '.\tickets-1000.json' -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 20
$hits = foreach ($t in $data.tickets) {
    if ($t.state -match '(?i)bloqu') {
        [pscustomobject]@{ id = $t.id; urgence = $t.expected.urgence; snippet = ([regex]::Match($t.state, '(?i).{0,25}bloqu.{0,25}')).Value }
    }
}
Write-Output ("tickets containing 'bloqu*': {0}" -f $hits.Count)
$hits | Group-Object urgence | ForEach-Object { "{0,4}  {1}" -f $_.Count, $_.Name }
$hits | Select-Object -First 6 | Format-Table -AutoSize
