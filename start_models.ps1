$ErrorActionPreference = "Continue"
lms server start
$loaded = (lms ps) -join "`n"
if ($loaded -match "nomic-embed-text") { Write-Host "Nomic already loaded" } else { lms load text-embedding-nomic-embed-text-v1.5 --gpu max }
if ($loaded -match "qwen3.5-9b") { Write-Host "Qwen already loaded" } else { lms load qwen/qwen3.5-9b --gpu max --context-length 16384 }
lms ps