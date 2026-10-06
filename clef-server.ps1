<#
.SYNOPSIS
  Lance llama-server avec Clef-flash (Q4_K_M) + support multimodal (mmproj).

.DESCRIPTION
  Modèle et binaire téléchargés dans .clef-cache/ (hors suivi git, ~7GB).
  Expose uniquement /v1/systemone (pas de chat/completion, pas d'UI web) -
  conforme à l'API Jev/SystemOne, un ticket/état par requête.

  Les modalités image/vidéo restent disponibles grâce à --mmproj : ce test
  porte sur des tickets texte, mais le serveur n'est pas restreint au texte.

.PARAMETER Port
  Port d'écoute (défaut 8711).

.PARAMETER ContextSize
  Taille du contexte (défaut 8192, largement suffisant pour un ticket).
#>
param(
    [int]$Port = 8711,
    [int]$ContextSize = 8192
)

$bin = Join-Path $PSScriptRoot ".clef-cache\bin\llama-server.exe"
$model = Join-Path $PSScriptRoot ".clef-cache\models\Cloudflare_clef-flash-Q4_K_M.gguf"
$mmproj = Join-Path $PSScriptRoot ".clef-cache\models\mmproj-Cloudflare_clef-flash-f16.gguf"

if (-not (Test-Path $bin)) { throw "llama-server.exe introuvable: $bin" }
if (-not (Test-Path $model)) { throw "Modèle introuvable: $model" }
if (-not (Test-Path $mmproj)) { throw "mmproj introuvable: $mmproj" }

& $bin -m $model --mmproj $mmproj --port $Port --host 127.0.0.1 -c $ContextSize
