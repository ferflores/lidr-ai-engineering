#!/usr/bin/env bash
# Verificación automática del proyecto. La usa el pipeline de CI y sirve también en local.
#
#   1. Estructura de carpetas y ficheros requeridos por el ejercicio (incluidas las plantillas
#      de prompt versionadas) y que el texto del prompt no está en el código.
#   2. Tests (pytest) con el LLM sustituido por dobles.
#   3. Arranque real del servicio: /health, /docs, /openapi.json, /api/v1/prompts, validación del
#      body tipado, y arranque del formulario de Streamlit.
#   4. (Opcional) Llamada real al LLM con parámetros si hay API key configurada.
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
  app/schemas/__init__.py app/schemas/estimation.py
  app/services/prompts.py
  prompts/README.md
  prompts/estimacion/v1/template.toml prompts/estimacion/v1/system.md.j2 prompts/estimacion/v1/user.md.j2
  prompts/estimacion/v2/template.toml prompts/estimacion/v2/system.md.j2 prompts/estimacion/v2/user.md.j2
  .env.example .gitignore pyproject.toml README.md
  streamlit_app.py
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
if grep -rn 'Eres un estimador' app streamlit_app.py >/dev/null 2>&1; then
  fail "el texto del prompt sigue dentro del código (debe vivir en prompts/)"
else
  ok "el texto del prompt no está en el código: vive en prompts/estimacion/<versión>/"
fi
if grep -rn 'st.chat_input\|st.chat_message' streamlit_app.py >/dev/null 2>&1; then
  fail "streamlit_app.py sigue usando el chat como interfaz"
else
  ok "streamlit_app.py no usa chat: la interfaz es un formulario"
fi
[ "$fallos" -eq 0 ] || { echo "Estructura incorrecta ($fallos fallos)"; exit 1; }

echo "== 2/4 Tests =="
uv run pytest -q

echo "== 3/4 Arranque de la API en $BASE_URL y del formulario de Streamlit =="
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
TYPED_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/v1/estimate" -H "Content-Type: application/json" -d '{"transcription": "El cliente quiere una landing page con formulario.", "estimation_mode": "agresiva"}')
[ "$TYPED_CODE" = "422" ] && ok "POST /api/v1/estimate rechaza un parámetro fuera de rango (422 con estimation_mode inválido)" || fail "parámetro inválido devolvió $TYPED_CODE"
VERSION_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/v1/estimate" -H "Content-Type: application/json" -d '{"transcription": "El cliente quiere una landing page con formulario.", "prompt_version": "v99"}')
[ "$VERSION_CODE" = "422" ] && ok "POST /api/v1/estimate rechaza una versión de plantilla inexistente (422)" || fail "versión inexistente devolvió $VERSION_CODE"
PROMPTS=$(curl -sf "$BASE_URL/api/v1/prompts") || { fail "GET /api/v1/prompts no responde"; PROMPTS=""; }
echo "$PROMPTS" | grep -q '"version":"v1"' && echo "$PROMPTS" | grep -q '"version":"v2"' && ok "GET /api/v1/prompts lista las plantillas v1 y v2" || fail "GET /api/v1/prompts no lista v1 y v2: $PROMPTS"

STREAMLIT_PORT=$((PORT + 1))
uv run streamlit run streamlit_app.py --server.headless true --server.port "$STREAMLIT_PORT" --server.address 127.0.0.1 >/dev/null 2>&1 &
STREAMLIT_PID=$!
trap 'kill "$SERVER_PID" "$STREAMLIT_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$STREAMLIT_PORT/_stcore/health" >/dev/null 2>&1 && break
  sleep 0.5
done
STREAMLIT_HEALTH=$(curl -sf "http://127.0.0.1:$STREAMLIT_PORT/_stcore/health" || true)
[ "$STREAMLIT_HEALTH" = "ok" ] && ok "streamlit run streamlit_app.py (formulario) arranca (GET /_stcore/health -> ok)" || fail "Streamlit no responde en el puerto $STREAMLIT_PORT"

echo "== 4/4 Llamada real al LLM =="
if [ "${SKIP_LLM:-0}" = "1" ]; then
  echo "  ⏭  omitida (SKIP_LLM=1)"
elif echo "$HEALTH" | grep -q '"llm_configured":true'; then
  RESPUESTA=$(scripts/estimar.sh transcripciones/reunion-landing-page-parametros.json "$BASE_URL")
  if echo "$RESPUESTA" | grep -q '"estimation"'; then
    ok "el endpoint devolvió una estimación real"
    RESPUESTA="$RESPUESTA" uv run python - <<'PY'
import json, os
respuesta = json.loads(os.environ["RESPUESTA"])
usage = respuesta.get("usage") or {}
cost = usage.get("cost") or {}
coste = f"{cost['total_usd']:.6f} USD" if cost else "desconocido"
prompt = respuesta.get("prompt") or {}
print(f"     modelo={respuesta['model']}  proveedor={respuesta['provider']}  plantilla={prompt.get('template')}/{prompt.get('version')}")
print(f"     tokens: {usage.get('input_tokens')} entrada + {usage.get('output_tokens')} salida = {usage.get('total_tokens')}  ·  coste: {coste}")
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
