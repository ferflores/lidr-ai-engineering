#!/usr/bin/env bash
# Verificación automática del proyecto. La usa el pipeline de CI y sirve también en local.
#
#   1. Estructura de carpetas y ficheros requeridos por el ejercicio (incluidas las plantillas
#      de prompt versionadas) y que el texto del prompt no está en el código.
#   2. Tests (pytest) con el LLM sustituido por dobles.
#   3. Arranque real del servicio: /health, /docs, /openapi.json, /api/v1/prompts, validación del
#      body tipado, sesiones (POST /api/v1/sessions, 404, multipart 422/415) y arranque del formulario.
#   4. (Opcional) Llamada real al LLM si hay API key: una estimación suelta y una conversación de dos
#      turnos con adjunto (SKIP_SESSION_LLM=1 para omitir solo esta última).
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
  app/services/sessions.py app/services/attachments.py app/services/metadata_extractor.py
  app/services/conversation.py app/routers/sessions.py app/clients/estimator_api.py
  prompts/README.md
  prompts/estimacion/v3/template.toml prompts/estimacion/v3/system.md.j2 prompts/estimacion/v3/user.md.j2
  transcripciones/panaderia-turno-2-cambios.txt transcripciones/panaderia-turno-3-adjunto.txt
  scripts/generar-adjuntos.py scripts/estimar-sesion.sh
  prompts/estimacion/v1/template.toml prompts/estimacion/v1/system.md.j2 prompts/estimacion/v1/user.md.j2
  prompts/estimacion/v2/template.toml prompts/estimacion/v2/system.md.j2 prompts/estimacion/v2/user.md.j2
  app/services/anchors.py app/services/summarizer.py app/services/tiering.py app/services/response_cache.py app/services/observability.py
  prompts/estimacion/v4/template.toml prompts/estimacion/v4/system.md.j2 prompts/estimacion/v4/user.md.j2
  prompts/resumen/v1/template.toml prompts/resumen/v1/system.md.j2 prompts/resumen/v1/user.md.j2
  evals/__init__.py evals/metrics.py evals/run.py evals/golden_dataset.json
  evals/stress/scenarios.py evals/stress/metrics.py evals/stress/run.py evals/stress/report.py evals/stress/fixtures/build_pdfs.py
  evals/stress/REPORT.md evals/stress/results.csv
  tests/test_stress_metrics.py
  app/embedding_pipeline/__init__.py app/embedding_pipeline/schemas.py app/embedding_pipeline/chunker.py
  app/embedding_pipeline/embedder.py app/embedding_pipeline/router.py app/embedding_pipeline/similarity.py
  app/embedding_pipeline/SANITY_CHECK.md scripts/compare.py data/budgets_sample.json
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
echo "$PROMPTS" | grep -q '"version":"v1"' && echo "$PROMPTS" | grep -q '"version":"v2"' && echo "$PROMPTS" | grep -q '"version":"v3"' && echo "$PROMPTS" | grep -q '"version":"v4"' && ok "GET /api/v1/prompts lista las plantillas v1, v2, v3 y v4" || fail "GET /api/v1/prompts no lista v1-v4: $PROMPTS"

# Sesiones conversacionales (sesión 05)
SESSION=$(curl -s -X POST "$BASE_URL/api/v1/sessions") || SESSION=""
SESSION_ID=$(echo "$SESSION" | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
[ -n "$SESSION_ID" ] && ok "POST /api/v1/sessions crea una sesión ($SESSION)" || fail "POST /api/v1/sessions no devolvió session_id: $SESSION"
STATE_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/api/v1/sessions/$SESSION_ID")
[ "$STATE_CODE" = "200" ] && ok "GET /api/v1/sessions/{id} -> 200" || fail "GET /api/v1/sessions/{id} -> $STATE_CODE"
NOT_FOUND_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/v1/sessions/no-existe/estimate" -F "transcript=El cliente quiere una landing page con formulario.")
[ "$NOT_FOUND_CODE" = "404" ] && ok "POST /api/v1/sessions/{id}/estimate con sesión inexistente -> 404" || fail "sesión inexistente devolvió $NOT_FOUND_CODE"
MP_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/v1/sessions/$SESSION_ID/estimate" -F "transcript=corta")
[ "$MP_CODE" = "422" ] && ok "POST /api/v1/sessions/{id}/estimate valida el multipart (422 con transcripción corta)" || fail "multipart inválido devolvió $MP_CODE"
ATT_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/api/v1/sessions/$SESSION_ID/estimate" -F "transcript=El cliente quiere una landing page con formulario." -F "attachments=@pyproject.toml;filename=programa.exe")
[ "$ATT_CODE" = "415" ] && ok "POST /api/v1/sessions/{id}/estimate rechaza un adjunto no soportado (415)" || fail "adjunto no soportado devolvió $ATT_CODE"
STATE=$(curl -s "$BASE_URL/api/v1/sessions/$SESSION_ID")
echo "$STATE" | grep -q '"last_turn_observed"' && echo "$STATE" | grep -q '"anchors_count"' && ok "GET /api/v1/sessions/{id} expone anchors_count, summary_chars y last_turn_observed" || fail "GET /api/v1/sessions/{id} sin los campos de observación: $STATE"

# Pipeline de embeddings (sesión 07): el endpoint está publicado y valida el body sin gastar créditos
curl -sf "$BASE_URL/openapi.json" | grep -q '"/embeddings/ingest"' && ok "POST /embeddings/ingest está en el OpenAPI" || fail "POST /embeddings/ingest no está en el OpenAPI"
EMB_EMPTY=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/embeddings/ingest" -H "Content-Type: application/json" -d '{}')
[ "$EMB_EMPTY" = "422" ] && ok "POST /embeddings/ingest valida el body (422 sin presupuestos)" || fail "body vacío en /embeddings/ingest devolvió $EMB_EMPTY"
EMB_BAD=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/embeddings/ingest" -H "Content-Type: application/json" -d '{"budgets":[{"budget_id":"X-1","client_metadata":{"name":"n","sector":"banking","country":"ES"},"project_summary":"s","main_technology":"t","year":2024,"total_estimated_hours":1,"components":[{"component_id":"C-1","name":"n","description":"d","tech_stack":[],"estimated_hours":1,"complexity":"high","dependencies":[]}]}]}')
[ "$EMB_BAD" = "422" ] && ok "POST /embeddings/ingest rechaza un sector fuera del vocabulario (422)" || fail "sector inválido en /embeddings/ingest devolvió $EMB_BAD"
# El chunker funciona sin tocar la API de embeddings: 15 presupuestos del sample -> 73 chunks trazables
if uv run python -c "
import json
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.schemas import Budget
budgets = [Budget.model_validate(b) for b in json.load(open('data/budgets_sample.json'))]
chunks = JSONStructuralChunker().chunk(budgets)
assert len(budgets) == 15, len(budgets)
assert len(chunks) == sum(len(b.components) for b in budgets) == 73, len(chunks)
assert chunks[0].chunk_id == 'BUD-2024-014::AUTH-001'
assert chunks[0].text.startswith('[Project: ') and chunks[0].token_count > 0
assert chunks[0].metadata['client_sector'] == 'finance'
" >/dev/null 2>&1; then
  ok "el chunker produce 73 chunks trazables de los 15 presupuestos del sample"
else
  fail "el chunker no produce los chunks esperados sobre data/budgets_sample.json"
fi

# Evals y stress (sesión 06) con el LLM simulado: el runner funciona de punta a punta sin API key
# (evals/run.py devuelve 1 si algún caso falla una métrica, y con el doble el recall falla siempre: aquí solo se comprueba que corre)
EVALS_OUT=$(uv run python -m evals.run --fake --cases landing-hubspot 2>/dev/null || true)
echo "$EVALS_OUT" | grep -q "casos pasan todas las métricas" && ok "evals/run.py corre de punta a punta con el LLM simulado" || fail "evals/run.py no corre con el LLM simulado: $EVALS_OUT"
STRESS_OUT=$(mktemp -d)
if uv run python -m evals.stress.run --fake --scenarios growing --attachment-sizes 0,5 --repeats 1 --turns 3 --output "$STRESS_OUT/results.csv" --report "$STRESS_OUT/REPORT.md" >/dev/null 2>&1 && [ "$(wc -l < "$STRESS_OUT/results.csv")" -eq 7 ] && grep -q "## 6. Lectura" "$STRESS_OUT/REPORT.md"; then
  ok "evals/stress/run.py produce el CSV (una fila por turno) y REPORT.md con el LLM simulado"
else
  fail "evals/stress/run.py no produce el CSV y el reporte esperados"
fi

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
  echo "  -- ingesta real de embeddings (15 presupuestos del sample) --"
  # La respuesta lleva 73 vectores de 1536 floats (~2 MB): va a un fichero temporal, no a una variable
  # de entorno ni a la línea de comandos (el límite de argumentos del sistema no lo admite).
  EMB_FILE=$(mktemp)
  curl -sf -X POST "$BASE_URL/embeddings/ingest" -H "Content-Type: application/json" \
    -d "{\"budgets\": $(cat data/budgets_sample.json)}" -o "$EMB_FILE" || true
  if grep -q '"estimated_cost_usd"' "$EMB_FILE" 2>/dev/null; then
    ok "POST /embeddings/ingest devolvió los chunks vectorizados"
    EMB_FILE="$EMB_FILE" uv run python - <<'PY'
import json, math, os
data = json.load(open(os.environ["EMB_FILE"]))
stats, chunks = data["stats"], data["chunks"]
print(f"     {stats['total_budgets']} presupuestos -> {stats['total_chunks']} chunks · {stats['dimensions']} dims · "
      f"{stats['total_tokens']} tokens ({stats['counted_tokens']} contados con tiktoken) · "
      f"{stats['estimated_cost_usd']:.6f} USD · {stats['batches']} lote(s) · {stats['latency_ms']} ms")
tokens = sorted(c["token_count"] for c in chunks)
largest = max(chunks, key=lambda c: c["token_count"])
norm = math.sqrt(sum(x * x for x in chunks[0]["embedding"]))
print(f"     tokens por chunk: min={tokens[0]} p50={tokens[len(tokens)//2]} max={tokens[-1]} ({largest['chunk_id']}) · "
      f"norma del primer vector: {norm:.4f}")
assert len(chunks[0]["embedding"]) == stats["dimensions"] == 1536
assert chunks[0]["chunk_id"] == "BUD-2024-014::AUTH-001"
PY
  else
    fail "la ingesta de embeddings no funcionó (respuesta en $EMB_FILE)"
  fi
  rm -f "$EMB_FILE"
  if [ "${SKIP_SESSION_LLM:-0}" = "1" ]; then
    echo "  ⏭  conversación omitida (SKIP_SESSION_LLM=1)"
  else
    echo "  -- conversación de dos turnos en una sesión (el segundo con la especificación en PDF) --"
    uv run scripts/generar-adjuntos.py >/dev/null
    SID=$(curl -sf -X POST "$BASE_URL/api/v1/sessions" | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
    T1=$(curl -sf -X POST "$BASE_URL/api/v1/sessions/$SID/estimate" -F "transcript=<transcripciones/panaderia-descripcion-detallada.txt" -F "project_name=Tienda online La Espiga" -F "detail_level=resumen") || T1=""
    T2=$(curl -sf -X POST "$BASE_URL/api/v1/sessions/$SID/estimate" -F "transcript=<transcripciones/panaderia-turno-3-adjunto.txt" -F "attachments=@transcripciones/adjuntos/panaderia-especificacion-tecnica.pdf;type=application/pdf" -F "detail_level=resumen") || T2=""
    if echo "$T2" | grep -q '"turn":2'; then
      ok "dos turnos en la misma sesión; el segundo con adjunto"
      T1="$T1" T2="$T2" uv run python - <<'PY'
import json, os
t1, t2 = json.loads(os.environ["T1"]), json.loads(os.environ["T2"])
print(f"     turno 1: {t1['window']['messages_sent']} mensajes al LLM · memoria: {t1['project_metadata']['project_name']!r}, {t1['project_metadata']['last_total_hours']} h")
print(f"     turno 2: {t2['window']['messages_sent']} mensajes al LLM · adjuntos: {[(a['filename'], a['chars']) for a in t2['attachments']]}")
print(f"     memoria tras el turno 2: {t2['project_metadata']['project_name']!r}, {t2['project_metadata']['last_total_hours']} h, tecnologías {t2['project_metadata']['mentioned_technologies']}")
assert t2["estimation"].startswith("## Estimación"), "el turno 2 no devolvió una estimación"
PY
    else
      fail "la conversación de dos turnos no funcionó: $T1 / $T2"
    fi
  fi
else
  echo "  ⏭  omitida (no hay API key configurada; rellena .env para probar el flujo completo)"
fi

echo
[ "$fallos" -eq 0 ] && echo "✅ Verificación completada sin fallos" || { echo "❌ Verificación con $fallos fallos"; exit 1; }
