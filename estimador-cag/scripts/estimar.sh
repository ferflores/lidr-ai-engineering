#!/usr/bin/env bash
# Envía una transcripción al endpoint POST /api/v1/estimate y muestra la respuesta.
#
# Uso:
#   scripts/estimar.sh                                          # transcripciones/reunion-landing-page.json
#   scripts/estimar.sh transcripciones/otra.json                # un cuerpo JSON ya preparado
#   scripts/estimar.sh transcripciones/otra.txt                 # texto plano: se envuelve en JSON
#   scripts/estimar.sh transcripciones/otra.json http://localhost:9000
set -euo pipefail
cd "$(dirname "$0")/.."

FILE="${1:-transcripciones/reunion-landing-page.json}"
BASE_URL="${2:-http://localhost:8000}"

case "$FILE" in
  *.json) BODY=$(cat "$FILE") ;;
  *)      BODY=$(uv run python -c 'import json, sys; print(json.dumps({"transcription": open(sys.argv[1], encoding="utf-8").read().strip()}, ensure_ascii=False))' "$FILE") ;;
esac

curl -sS -X POST "$BASE_URL/api/v1/estimate" \
  -H "Content-Type: application/json" \
  -d "$BODY"
echo
