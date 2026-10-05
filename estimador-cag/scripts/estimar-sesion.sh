#!/usr/bin/env bash
# Conversación de tres turnos contra la API con curl: crea una sesión, envía la transcripción detallada de la
# panadería, después los cambios de alcance y, por último, la especificación técnica adjunta en PDF.
# Tras cada turno imprime la memoria del proyecto (project_metadata) y el estado de la ventana.
#
# Uso:
#   scripts/estimar-sesion.sh                      # contra http://localhost:8000
#   scripts/estimar-sesion.sh http://localhost:9000
#   SOLO_TURNOS=1 scripts/estimar-sesion.sh        # solo el primer turno
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${1:-http://localhost:8000}"
ADJUNTOS=transcripciones/adjuntos
[ -f "$ADJUNTOS/panaderia-especificacion-tecnica.pdf" ] || uv run scripts/generar-adjuntos.py >/dev/null

resumen() {
  # Imprime la estimación y el estado de la sesión de la respuesta JSON que llega por stdin.
  # (Se guarda en una variable: el heredoc del script Python ocupa stdin.)
  RESPUESTA="$(cat)"
  RESPUESTA="$RESPUESTA" uv run python - <<'PY'
import json, os, sys
try:
    d = json.loads(os.environ["RESPUESTA"])
except json.JSONDecodeError:
    sys.exit(f"Respuesta no válida de la API: {os.environ['RESPUESTA'][:300]!r}")
if "estimation" not in d:
    sys.exit(f"Error de la API: {d}")
w = d["window"]
print(f"── Turno {d['turn']} · {w['messages_sent']} mensajes al LLM · ventana {w['turns_in_window']}/{w['max_turns']} turnos ({w['dropped_turns']} descartados)")
for a in d["attachments"]:
    print(f"   📎 {a['filename']}: {a['kind']}, {a['chars']} caracteres extraídos" + (f", {a['pages']} páginas" if a.get("pages") else "") + (f" ⚠ {a['warning']}" if a.get("warning") else ""))
u = d.get("usage") or {}
if u:
    coste = (u.get("cost") or {}).get("total_usd")
    print(f"   tokens: {u['input_tokens']} entrada + {u['output_tokens']} salida" + (f" · {coste:.6f} USD" if coste is not None else ""))
print()
print("\n".join("   " + l for l in d["estimation"].splitlines()))
print()
print("   project_metadata:")
for k, v in d["project_metadata"].items():
    if v not in (None, [], "") and k != "updated_at":
        print(f"     - {k}: {v}")
print()
PY
}

echo "== Creando sesión en $BASE_URL =="
SESSION=$(curl -sS -X POST "$BASE_URL/api/v1/sessions")
echo "$SESSION"
SESSION_ID=$(echo "$SESSION" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
URL="$BASE_URL/api/v1/sessions/$SESSION_ID/estimate"

echo
echo "== Turno 1: transcripción detallada de la panadería =="
curl -sS -X POST "$URL" \
  -F "transcript=<transcripciones/panaderia-descripcion-detallada.txt" \
  -F "project_name=Tienda online La Espiga" \
  -F "project_type=web" \
  -F "hourly_rate_eur=50" \
  -F "deadline_weeks=8" \
  -F "detail_level=resumen" | resumen

[ "${SOLO_TURNOS:-}" = "1" ] && exit 0

echo "== Turno 2: cambios de alcance (sin repetir la transcripción ni el nombre) =="
curl -sS -X POST "$URL" \
  -F "transcript=<transcripciones/panaderia-turno-2-cambios.txt" \
  -F "hourly_rate_eur=50" \
  -F "deadline_weeks=8" \
  -F "detail_level=resumen" | resumen

echo "== Turno 3: adjunta la especificación técnica (PDF) y la propuesta anterior (Word) =="
curl -sS -X POST "$URL" \
  -F "transcript=<transcripciones/panaderia-turno-3-adjunto.txt" \
  -F "attachments=@$ADJUNTOS/panaderia-especificacion-tecnica.pdf;type=application/pdf" \
  -F "attachments=@$ADJUNTOS/panaderia-propuesta-anterior.docx" \
  -F "hourly_rate_eur=50" \
  -F "deadline_weeks=8" \
  -F "detail_level=resumen" | resumen

echo "== Estado final de la sesión (GET /api/v1/sessions/$SESSION_ID) =="
ESTADO="$(curl -sS "$BASE_URL/api/v1/sessions/$SESSION_ID")"
ESTADO="$ESTADO" uv run python - <<'PY'
import json, os
d = json.loads(os.environ["ESTADO"])
print(f"turnos: {d['turn_count']} · ventana: {d['window']}")
for m in d["history"]:
    primera = m["content"].strip().splitlines()[0][:70]
    print(f"  {m['role']:9s} {m['chars']:6d} caracteres  {primera}")
PY
