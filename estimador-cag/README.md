# Estimador CAG — Proyecto 1 (Sesión 02)

Servicio **FastAPI** que recibe la transcripción de una reunión con un cliente y devuelve una
**estimación de software** (desglose de tareas, horas, coste, equipo y duración) generada por un LLM
(**OpenAI** o **Anthropic**).

Arquitectura **CAG**: todo el contexto que necesita el modelo (estimaciones de ejemplo) viaja íntegro
dentro del prompt en cada llamada. No hay base de datos, ni retrieval, ni persistencia.

## Cómo funciona

```
POST /api/v1/estimate  { "transcription": "..." }
        │
        ▼
app/routers/estimations.py ─ valida el body con Pydantic
        │
        ▼
app/services/llm_service.py ─ construye los mensajes:
        [system]  instrucciones + ejemplos de app/context/examples.py   ◄── CAG
        [user]    transcripción de la reunión
        │
        ▼
OpenAI (gpt-4o-mini)  ó  Anthropic (claude-haiku-4-5)      ◄── LLM_PROVIDER
        │
        ▼
{ "estimation": "## Estimación: ...", "model": "...", "provider": "...", "usage": {...}, "generated_at": "..." }
```

## Estructura

```
estimador-cag/
├── app/
│   ├── main.py                 # aplicación FastAPI, /health, Swagger
│   ├── config.py               # Settings (pydantic-settings) cargados desde .env
│   ├── routers/estimations.py  # POST /api/v1/estimate + schemas de request/response
│   ├── services/llm_service.py # system prompt + inyección de ejemplos + llamada al LLM
│   └── context/examples.py     # estimaciones de ejemplo (el "conocimiento" del sistema)
├── tests/                      # pytest: API, inyección de contexto y proveedores (LLM simulado)
├── scripts/
│   ├── estimar.sh              # envía una transcripción al endpoint con curl
│   ├── generar-json.sh         # crea transcripciones/*.json a partir de los .txt
│   └── verificar.sh            # estructura + tests + arranque real (lo usa el CI)
├── transcripciones/            # transcripciones de ejemplo (.txt legible, .json listo para curl)
├── docs/                       # comparativa de resultados (transcripción pobre vs. detallada)
├── Dockerfile                  # imagen basada en uv + Python 3.11
├── docker-compose.yml          # arranque con Docker, lee .env sin copiarlo a la imagen
├── .dockerignore
├── .env.example                # variables necesarias (sin valores)
├── .env                        # tus valores reales (ignorado por git y por docker)
├── pyproject.toml / uv.lock
└── README.md
```

## Requisitos

- [uv](https://docs.astral.sh/uv/) (instala solo el Python 3.11 fijado en `.python-version`), o bien
  [Docker Desktop](https://www.docker.com/get-started) para ejecutarlo en contenedor.
- Una API key de [OpenAI](https://platform.openai.com/) o de [Anthropic](https://console.anthropic.com/).

## Puesta en marcha

```bash
cd estimador-cag
uv sync                                  # crea .venv e instala dependencias
cp .env.example .env                     # rellena OPENAI_API_KEY o ANTHROPIC_API_KEY
uv run uvicorn app.main:app --reload     # http://localhost:8000
```

- Swagger UI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

## Ejecutar con Docker

```bash
cp .env.example .env                 # rellena la API key (solo la primera vez)
docker compose up --build -d         # construye la imagen y arranca en http://localhost:8000
docker compose logs -f api           # ver los logs
docker compose down                  # parar
```

Lanzar estimaciones por curl contra el contenedor:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/reunion-landing-page.json

curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/reunion-app-reservas-restaurante.json
```

Para desarrollar sobre el contenedor, `docker compose watch` sincroniza `app/` y reinicia el servicio
al guardar; si cambian `pyproject.toml` o `uv.lock`, reconstruye la imagen.

Cómo está montado:

- La imagen parte de `ghcr.io/astral-sh/uv:python3.11-bookworm-slim` e instala las dependencias con
  `uv sync --frozen --no-dev` en una capa aparte, así solo se reinstalan si cambia `uv.lock`.
- El `.env` **no se copia a la imagen** (está en `.dockerignore`): `docker compose` lo inyecta como
  variables de entorno al arrancar. Sin `.env` el servicio arranca igualmente, pero el endpoint de
  estimación responde `500` hasta que se configure la API key.
- El contenedor corre con un usuario sin privilegios y tiene un `HEALTHCHECK` sobre `/health`.

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `LLM_PROVIDER` | Proveedor a usar: `openai` o `anthropic` | `openai` |
| `OPENAI_API_KEY` | API key de OpenAI (si `LLM_PROVIDER=openai`) | — |
| `ANTHROPIC_API_KEY` | API key de Anthropic (si `LLM_PROVIDER=anthropic`) | — |
| `OPENAI_MODEL` | Modelo de OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Modelo de Anthropic | `claude-haiku-4-5` |
| `LLM_TEMPERATURE` | Temperatura del modelo (0-2) | `0.2` |
| `LLM_MAX_TOKENS` | Máximo de tokens de salida | `4096` |
| `LLM_TIMEOUT_SECONDS` | Timeout de la llamada al proveedor | `60` |

Las API keys solo viven en `.env`, que está en `.gitignore`. Nunca aparecen en el código.

## Probar el endpoint

Con el script (usa `transcripciones/reunion-landing-page.json` por defecto; acepta `.json` o `.txt`):

```bash
scripts/estimar.sh
scripts/estimar.sh transcripciones/reunion-app-reservas-restaurante.json
```

Con curl y un cuerpo ya preparado:

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/reunion-landing-page.json
```

Con curl y el texto inline:

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas. El diseño ya existe en Figma."
  }'
```

Respuesta:

```json
{
  "estimation": "## Estimación: Landing Page con Blog e Integración HubSpot\n\n### Resumen del alcance\n...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "usage": { "input_tokens": 2100, "output_tokens": 620, "total_tokens": 2720 },
  "generated_at": "2026-09-16T10:30:00.000000Z"
}
```

Códigos de respuesta: `200` estimación generada · `422` body inválido · `500` falta la API key · `502` el proveedor ha fallado.

## Comparar una transcripción pobre con una detallada

La calidad de la estimación depende directamente de la calidad de la transcripción. Para verlo hay dos
transcripciones del **mismo proyecto**, una tienda online para una panadería artesanal:

| Archivo | Qué contiene |
|---|---|
| `transcripciones/panaderia-descripcion-pobre.json` | Tres frases vagas: quiere vender por internet, que esté lista pronto y que no sea cara. |
| `transcripciones/panaderia-descripcion-detallada.json` | Reunión completa: catálogo, stock diario, franjas de recogida, reparto, pagos, pedidos recurrentes, panel del obrador, idiomas, plazo y presupuesto. |

```bash
# 1) Levantar la app con Docker
cd estimador-cag
docker compose up --build -d
curl http://localhost:8000/health

# 2) Transcripción pobre
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/panaderia-descripcion-pobre.json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["estimation"]); print(d["usage"])'

# 3) Transcripción detallada
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/panaderia-descripcion-detallada.json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["estimation"]); print(d["usage"])'

# 4) Parar
docker compose down
```

El `| python3 ...` solo imprime la estimación legible; sin él se ve el JSON completo.

Resultado obtenido el 2026-09-16 con `gpt-4o-mini` y temperatura 0.2. Salida completa y análisis en
[`docs/comparativa-pobre-vs-detallada.md`](docs/comparativa-pobre-vs-detallada.md).

| | Pobre | Detallada |
|---|---|---|
| Caracteres de entrada | 173 | 2.672 |
| Tokens (entrada / salida) | 1.929 / 341 | 2.581 / 431 |
| Tareas | 7 genéricas | 10 específicas (pedidos recurrentes, panel del obrador, notificaciones, hosting) |
| Total | 230 h · 11.500 € · 6-8 semanas | 410 h · 20.500 € · 10-12 semanas |
| Supuestos | rellenan lo que no se dijo | recogen lo que sí se dijo (sin TPV ni contabilidad) |
| Preguntas | básicas (métodos de pago, registro) | concretas (lista de producción, exportación a Excel) |

Qué observar:

- Con la transcripción pobre el modelo inventa un alcance estándar y lo estima: la cifra parece
  razonable, pero no describe el proyecto real.
- Con la detallada el desglose refleja lo hablado, pero todavía se deja cosas: no avisa de que supera el
  presupuesto de 12.000 € y el plazo de 8 semanas, y omite el reparto por radio, el bilingüismo, Bizum y
  la exportación a Excel. Es el punto de partida para iterar el prompt en la sesión en vivo.

Para añadir un caso nuevo: escribe la transcripción en `transcripciones/<nombre>.txt`, ejecuta
`scripts/generar-json.sh` para crear el `.json` equivalente, lánzalo con curl o con
`scripts/estimar.sh transcripciones/<nombre>.json`, y anota el resultado en
`docs/comparativa-pobre-vs-detallada.md`.

## Tests y verificación automática

```bash
uv run pytest -q          # tests con el LLM sustituido por dobles (no gasta créditos)
scripts/verificar.sh      # estructura + tests + arranque real + llamada real si hay API key
```

`scripts/verificar.sh` comprueba, en orden:

1. Que existen todos los ficheros de la estructura del ejercicio, que `.env` está en `.gitignore` y no
   versionado, y que no hay API keys en el código.
2. Que pasan los tests.
3. Que el servicio arranca y responden `GET /health` (200), `GET /docs` (200), que `POST /api/v1/estimate`
   está en el OpenAPI y valida el body (422).
4. Si hay API key configurada, que `POST /api/v1/estimate` devuelve una estimación real.

El mismo script lo ejecuta el pipeline de GitHub Actions (`.github/workflows/estimador-cag.yml`, en la raíz
del repositorio) en cada push que toque `estimador-cag/`. Un segundo job construye la imagen Docker, la
arranca con `docker compose` y comprueba `/health`. Si se configuran los secrets `OPENAI_API_KEY` o
`ANTHROPIC_API_KEY` en el repositorio, el pipeline también hace la llamada real al LLM.

## Checklist del ejercicio

- [x] El proyecto arranca sin errores con `uv run uvicorn app.main:app --reload`
- [x] Las API keys se cargan desde `.env` y nunca aparecen en el código
- [x] `GET /health` responde con status 200
- [x] `POST /api/v1/estimate` recibe una transcripción y devuelve una estimación
- [x] La estimación se inspira en los ejemplos de contexto inyectados (mismo formato, magnitudes similares)
- [x] La documentación Swagger está accesible en `/docs`
- [x] `.env` está en `.gitignore`

## Decisiones de diseño

- **Dos proveedores**, seleccionables con `LLM_PROVIDER`. Ambos usan los modelos económicos que indica el
  curso. La estructura de mensajes es la misma: `[system]` instrucciones + ejemplos, `[user]` transcripción.
- **El contexto se inyecta en el system prompt** (`build_system_prompt`), no en el mensaje de usuario:
  así el modelo lo trata como instrucciones estables y la transcripción como el dato a procesar.
- **El system prompt fija un formato de salida** idéntico al de los ejemplos, para que las estimaciones
  nuevas sean comparables con las históricas.
- **SDK de Anthropic 1.x**: `temperature` ya no es un argumento de `messages.create`; para `claude-haiku-4-5`
  se envía en `extra_body`. Si se cambia a Sonnet 5 u Opus 4.7+, hay que quitarlo (esos modelos lo rechazan).
- **Errores traducidos a HTTP**: falta de configuración → `500`; error del proveedor (auth, cuota, red,
  respuesta vacía, rechazo) → `502`. Los detalles llegan en el campo `detail`.
- **Campos extra en la respuesta**: `usage` (tokens) y `generated_at`, útiles para medir coste y depurar.
- **Docker opcional**: el ejercicio no lo pide, pero el stack del curso corre en contenedores. El
  `Dockerfile` y el `docker-compose.yml` no cambian nada del código: la misma app corre con `uv run` o
  en contenedor.

## Próximos pasos (sesión en vivo)

- Mejorar los ejemplos de contexto (más variedad, casos parecidos a los reales).
- Iterar el prompt y comparar la calidad de las estimaciones entre proveedores.
- Explorar parámetros del modelo (`LLM_TEMPERATURE`, `LLM_MAX_TOKENS`).
- Evolucionar de CAG a RAG cuando los ejemplos no quepan en el contexto.
