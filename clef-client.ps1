<#
.SYNOPSIS
  Client PowerShell pour tester Clef-flash (Cloudflare) via llama-server / /v1/systemone.

.DESCRIPTION
  Contrairement à Laya (endpoint batch /v1/systemone/batch), llama-server n'expose
  qu'un endpoint /v1/systemone traitant UN ticket par requête. Ce script boucle
  donc sur chaque ticket du fichier d'entrée et envoie une requête séquentielle.

  Prérequis : le serveur llama-server doit déjà tourner, par exemple :
    .\.clef-cache\bin\llama-server.exe `
      -m .clef-cache\models\Cloudflare_clef-flash-Q4_K_M.gguf `
      --mmproj .clef-cache\models\mmproj-Cloudflare_clef-flash-f16.gguf `
      --port 8711 --host 127.0.0.1 -c 8192

.PARAMETER TicketsFile
  Fichier JSON contenant {questions, tickets:[{id, state, expected}]}.
  Par défaut: tickets-503-laya-heldout.json (le jeu de test honnête de Laya).

.PARAMETER ServerUrl
  URL du endpoint /v1/systemone (défaut: http://127.0.0.1:8711/v1/systemone).

.PARAMETER Limit
  Nombre de tickets à traiter (défaut: tous). Utile pour un test rapide.

.PARAMETER OutFile
  Fichier JSON de sortie avec le détail des réponses et un résumé d'exactitude.

.EXAMPLE
  # Test léger (10 tickets)
  .\clef-client.ps1 -Limit 10

.EXAMPLE
  # Test complet sur les 503 tickets held-out, résultats sauvegardés
  .\clef-client.ps1 -OutFile resultats-clef-flash-503.json
#>
param(
    [string]$TicketsFile = "tickets-503-laya-heldout.json",
    [string]$ServerUrl = "http://127.0.0.1:8711/v1/systemone",
    [int]$Limit = 0,
    [string]$OutFile = ""
)

$ErrorActionPreference = "Stop"

$data = Get-Content $TicketsFile -Raw -Encoding UTF8 | ConvertFrom-Json
$questions = $data.questions
$tickets = $data.tickets
if ($Limit -gt 0) { $tickets = $tickets[0..([Math]::Min($Limit, $tickets.Count) - 1)] }

$total = $tickets.Count
$okService = 0
$okRemb = 0
$results = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

for ($i = 0; $i -lt $tickets.Count; $i++) {
    $t = $tickets[$i]
    $body = @{ state = $t.state; questions = $questions } | ConvertTo-Json -Depth 10
    $resp = Invoke-RestMethod -Uri $ServerUrl -Method Post -Body $body -ContentType "application/json; charset=utf-8"

    $gotService = $resp.answers.service.choice
    $gotRemb = $resp.answers.remboursement.noul -gt 0.5
    $gotUrgScore = $resp.answers.urgence.score

    $serviceMatch = ($gotService -eq $t.expected.service)
    $rembMatch = ($gotRemb -eq $t.expected.remboursement)
    if ($serviceMatch) { $okService++ }
    if ($rembMatch) { $okRemb++ }

    $results += [PSCustomObject]@{
        id               = $t.id
        expected_service = $t.expected.service
        got_service      = $gotService
        service_match    = $serviceMatch
        expected_remb    = $t.expected.remboursement
        got_remb         = $gotRemb
        remb_match       = $rembMatch
        expected_urgence = $t.expected.urgence
        got_urgence_score = $gotUrgScore
    }

    if (($i + 1) % 25 -eq 0 -or ($i + 1) -eq $total) {
        Write-Host ("[{0}/{1}] service={2:P1} remboursement={3:P1} ({4:N0}s)" -f ($i + 1), $total, ($okService / ($i + 1)), ($okRemb / ($i + 1)), $sw.Elapsed.TotalSeconds)
    }
}

Write-Host ""
Write-Host "=== Résumé Clef-flash (Q4_K_M) ==="
Write-Host ("Tickets testés        : {0}" -f $total)
Write-Host ("Exactitude service     : {0:P2} ({1}/{2})" -f ($okService / $total), $okService, $total)
Write-Host ("Exactitude remboursement: {0:P2} ({1}/{2})" -f ($okRemb / $total), $okRemb, $total)
Write-Host ("Durée totale            : {0:N1}s ({1:N2}s/ticket)" -f $sw.Elapsed.TotalSeconds, ($sw.Elapsed.TotalSeconds / $total))

if ($OutFile) {
    $summary = [PSCustomObject]@{
        model               = "clef-flash-Q4_K_M"
        total               = $total
        accuracy_service    = $okService / $total
        accuracy_remboursement = $okRemb / $total
        duration_seconds    = $sw.Elapsed.TotalSeconds
        results             = $results
    }
    $summary | ConvertTo-Json -Depth 10 | Set-Content -Path $OutFile -Encoding UTF8
    Write-Host ("Résultats détaillés écrits dans {0}" -f $OutFile)
}
