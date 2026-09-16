#!/usr/bin/env bash
# Envía una transcripción al endpoint POST /api/v1/estimate y muestra la respuesta.
#
# Uso:
#   scripts/estimar.sh                                   # usa transcripciones/reunion-landing-page.txt
#   scripts/estimar.sh transcripciones/otra.txt          # otra transcripción
#   scripts/estimar.sh transcripciones/otra.txt http://localhost:9000
set -euo pipefail
cd "$(dirname "$0")/.."

FILE="${1:-transcripciones/reunion-landing-page.txt}"
BASE_URL="${2:-http://localhost:8000}"

BODY=$(uv run python -c 'import json, sys; print(json.dumps({"transcription": open(sys.argv[1], encoding="utf-8").read()}, ensure_ascii=False))' "$FILE")

curl -sS -X POST "$BASE_URL/api/v1/estimate" \
  -H "Content-Type: application/json" \
  -d "$BODY"
echo
