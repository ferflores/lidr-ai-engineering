#!/usr/bin/env bash
# Verificación automática del proyecto. La usa el pipeline de CI y sirve también en local.
#
#   1. Estructura de carpetas y ficheros requeridos por el ejercicio.
#   2. Tests (pytest) con el LLM sustituido por dobles.
#   3. Arranque real del servicio: /health, /docs y /openapi.json.
#   4. (Opcional) Llamada real al LLM si hay API key configurada.
#
# Uso: scripts/verificar.sh          (variables: PORT, SKIP_LLM=1)
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8765}"
BASE_URL="http://127.0.0.1:$PORT"
fallos=0

ok()   { echo "  ✅ $*"; }
fail() { echo "  ❌ $*"; fallos=$((fallos + 1)); }

echo "== 1/4 Estructura del proyecto =="
REQUIRED=(
  app/__init__.py app/main.py app/config.py
  app/routers/__init__.py app/routers/estimations.py
  app/services/__init__.py app/services/llm_service.py
  app/context/__init__.py app/context/examples.py
  .env.example .gitignore pyproject.toml README.md
)
for f in "${REQUIRED[@]}"; do
  [ -e "$f" ] && ok "$f" || fail "falta $f"
done
grep -qx '\.env' .gitignore && ok ".env está en .gitignore" || fail ".env no está en .gitignore"
if git ls-files --error-unmatch .env >/dev/null 2>&1; then fail ".env está versionado en git"; else ok ".env no está versionado"; fi
if grep -rnE 'sk-(ant-|proj-)?[A-Za-z0-9_-]{20,}' app scripts tests 2>/dev/null; then
  fail "hay algo que parece una API key en el código"
else
  ok "ninguna API key en el código"
fi
[ "$fallos" -eq 0 ] || { echo "Estructura incorrecta ($fallos fallos)"; exit 1; }

echo "== 2/4 Tests =="
uv run pytest -q

echo "== 3/4 Arranque del servicio en $BASE_URL =="
uv run uvicorn app.main:app --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do
  curl -sf "$BASE_URL/health" >/dev/null 2>&1 && break
  sleep 0.5
done
HEALTH=$(curl -sf "$BASE_URL/health") || { fail "GET /health no responde"; exit 1; }
echo "$HEALTH" | grep -q '"status":"ok"' && ok "GET /health -> $HEALTH" || fail "GET /health sin status ok: $HEALTH"
DOCS_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/docs")
[ "$DOCS_CODE" = "200" ] && ok "GET /docs -> 200" || fail "GET /docs -> $DOCS_CODE"
curl -sf "$BASE_URL/openapi.json" | grep -q '"/api/v1/estimate"' && ok "POST /api/v1/estimate está en el OpenAPI" || fail "POST /api/v1/estimate no está en el OpenAPI"
VALIDATION_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/v1/estimate" -H "Content-Type: application/json" -d '{"transcription": ""}')
[ "$VALIDATION_CODE" = "422" ] && ok "POST /api/v1/estimate valida el body (422 con transcripción vacía)" || fail "validación del body devolvió $VALIDATION_CODE"

echo "== 4/4 Llamada real al LLM =="
if [ "${SKIP_LLM:-0}" = "1" ]; then
  echo "  ⏭  omitida (SKIP_LLM=1)"
elif echo "$HEALTH" | grep -q '"llm_configured":true'; then
  RESPUESTA=$(scripts/estimar.sh transcripciones/reunion-landing-page.json "$BASE_URL")
  if echo "$RESPUESTA" | grep -q '"estimation"'; then
    ok "el endpoint devolvió una estimación real"
    RESPUESTA="$RESPUESTA" uv run python - <<'PY'
import json, os
respuesta = json.loads(os.environ["RESPUESTA"])
print(f"     modelo={respuesta['model']}  proveedor={respuesta['provider']}  tokens={respuesta.get('usage')}")
print()
for linea in respuesta["estimation"].splitlines():
    print("     " + linea)
PY
  else
    fail "el endpoint no devolvió una estimación: $RESPUESTA"
  fi
else
  echo "  ⏭  omitida (no hay API key configurada; rellena .env para probar el flujo completo)"
fi

echo
[ "$fallos" -eq 0 ] && echo "✅ Verificación completada sin fallos" || { echo "❌ Verificación con $fallos fallos"; exit 1; }
