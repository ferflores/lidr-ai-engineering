# Estimador CAG — Proyecto 1 (Sesiones 02 a 07)

Servicio **FastAPI** que recibe la transcripción de una reunión con un cliente más unos **parámetros
tipados** (tarifa, tipo de proyecto, plazo, enfoque, nivel de detalle) y devuelve una **estimación de
software** (desglose de tareas, horas, coste, equipo y duración) generada por un LLM (**OpenAI** o
**Anthropic**). Los prompts son **plantillas versionadas** en `prompts/`, fuera del código.

Desde el pre-work de la **sesión 05** el estimador deja de ser solo transaccional: una **sesión**
(`POST /api/v1/sessions`) encadena varios turnos sobre el mismo proyecto con **memoria conversacional**
(historial con ventana deslizante + `project_metadata` inyectado en el system prompt) y acepta
**adjuntos** PDF, Word y texto cuyo contenido se extrae localmente. La interfaz es un **formulario en
Streamlit** que ahora habla con la API por HTTP y muestra la memoria y el historial efectivo en un panel.
Ver [Sesiones conversacionales con memoria y adjuntos](#sesiones-conversacionales-con-memoria-y-adjuntos-pre-work-sesión-05).

Desde el directo de la **sesión 05** y el pre-work de la **sesión 06** el historial se **comprime** (los
turnos que salen de la ventana se pliegan en un **resumen acumulativo** y los hechos del cliente se
conservan como **anclas**), cada turno resuelve un **tier**, pasa por una **caché** exacta y semántica y
deja un único evento **`turn_observed`** con sus 13 campos; sobre eso hay un **framework de evals**
(`evals/`) y un **stress** del CAG (`evals/stress/`) cuyo resultado es [`evals/stress/REPORT.md`](evals/stress/REPORT.md).
Ver [Compresión, tier, caché y observación](#compresión-del-historial-tier-caché-y-observación-sesión-05-directo)
y [Stress del CAG](#stress-del-cag-pre-work-sesión-06).

Desde el pre-work de la **sesión 07** hay un primer paso hacia la búsqueda semántica: el módulo
`app/embedding_pipeline/` convierte presupuestos históricos en JSON en **chunks vectorizados**
(`POST /embeddings/ingest`), con un chunker estructural (un componente = un chunk) y
`text-embedding-3-small`. Los vectores se devuelven por HTTP, todavía **sin persistir**.
Ver [Pipeline de embeddings](#pipeline-de-embeddings-pre-work-sesión-07).

Además, cuatro piezas que el curso enseñó y el proyecto no tenía (ver
[Huecos del curso, cerrados](#huecos-del-curso-cerrados)): **salida estructurada** validada con Pydantic,
**guardrails** de entrada y salida, **caché semántica por embeddings** y el wrapper de proveedores con
**LiteLLM** como backend opcional.

Arquitectura **CAG**: todo el contexto que necesita el modelo (estimaciones de ejemplo) viaja íntegro
dentro del prompt en cada llamada. No hay base de datos ni retrieval; el único estado es el de las
sesiones, en memoria del proceso.

## Cómo funciona

```
POST /api/v1/estimate  { "transcription": "...", "hourly_rate_eur": 50, "deadline_weeks": 4, "estimation_mode": "conservadora", ... }
        │                                              ▲
        │                       streamlit_app.py ──────┘  el formulario construye el mismo EstimationRequest
        ▼
app/schemas/estimation.py ─ EstimationRequest: parámetros tipados (Pydantic), campos desconocidos rechazados
        │
        ▼
app/services/prompts.py ─ carga prompts/estimacion/<versión>/ (PROMPT_VERSION o prompt_version de la petición)
        │
        ▼
app/services/llm_service.py ─ renderiza la plantilla y construye los mensajes:
        [system]  system.md.j2: rol, reglas, formato + ejemplos de app/context/examples.py   ◄── CAG (fijo)
        [user]    user.md.j2: parámetros + transcripción                                       ◄── variable
        │
        ▼
OpenAI (gpt-4o-mini)  ó  Anthropic (claude-haiku-4-5)      ◄── LLM_PROVIDER
        │
        ▼
{ "estimation": "## Estimación: ...", "usage": {...}, "prompt": { "template": "estimacion", "version": "v3", ... } }
```

Y el flujo conversacional (pre-work sesión 05), donde la interfaz habla con la API por HTTP:

```
POST /api/v1/sessions                      → { "session_id": "<uuid4>", "max_turns": 6 }
POST /api/v1/sessions/{id}/estimate        multipart/form-data: transcript + parámetros + attachments[]
        │
        ▼
app/services/attachments.py ─ extrae el texto de PDF (pypdf), Word (python-docx) y .txt/.md y lo concatena
        │                     a la transcripción con separadores "--- attachment: <fichero> ---"
        ▼
app/services/conversation.py ─ un turno:
        system  = plantilla v3 con la MEMORIA del proyecto (<project_metadata>) al final      ◄── ProjectMetadata
        history = últimos N turnos de la sesión (ventana deslizante, N = SESSION_MAX_TURNS)    ◄── ConversationHistory
        user    = parámetros + transcripción (+ adjuntos)
        → LLM → estimación → se guarda el turno → metadata_extractor actualiza la memoria
        ▼
{ ...misma respuesta que /estimate..., "session_id", "turn", "window": {...}, "project_metadata": {...}, "attachments": [...] }
```

## Estructura

```
estimador-cag/
├── app/
│   ├── main.py                 # aplicación FastAPI, /health, Swagger
│   ├── config.py               # Settings (pydantic-settings) cargados desde .env, incluida PROMPT_VERSION
│   ├── schemas/estimation.py   # EstimationRequest: los parámetros tipados (contrato de API y formulario)
│   ├── routers/estimations.py  # POST /api/v1/estimate, GET /api/v1/prompts + schemas de respuesta
│   ├── routers/sessions.py     # POST /api/v1/sessions, GET|DELETE /sessions/{id}, POST /sessions/{id}/estimate (multipart)
│   ├── services/prompts.py     # registro de plantillas: carga prompts/<nombre>/<versión>/ y las renderiza
│   ├── services/llm_service.py # renderiza la plantilla (con memoria e historial) y llama al LLM (sin texto de prompt)
│   ├── services/sessions.py    # ConversationHistory (ventana deslizante), ProjectMetadata (memoria), Session, SessionStore
│   ├── services/attachments.py # extracción local de texto de PDF / Word / texto (camino B) y concatenación al transcript
│   ├── services/metadata_extractor.py  # heurística que actualiza project_metadata tras cada turno
│   ├── services/conversation.py        # orquesta un turno: adjuntos → compresión → prompt → caché/LLM → historial → memoria → turn_observed
│   ├── services/anchors.py     # anclas heurísticas: hechos del cliente que sobreviven a la ventana (bloque <anchors>)
│   ├── services/summarizer.py  # resumen acumulativo de los turnos que salen de la ventana (bloque <summary>; LLM o heurístico)
│   ├── services/tiering.py     # tier dinámico: modelo y tokens de salida según el contexto del turno
│   ├── services/response_cache.py      # caché de respuestas exacta y semántica (Jaccard), en memoria del proceso
│   ├── services/observability.py       # el evento structlog `turn_observed` (13 campos + extras)
│   ├── services/pricing.py     # precios por millón de tokens y cálculo del coste de cada llamada
│   ├── clients/estimator_api.py        # cliente HTTP de la API que usa la interfaz de Streamlit
│   ├── context/examples.py     # estimaciones de ejemplo (el "conocimiento" del sistema)
│   └── embedding_pipeline/     # pre-work sesión 07: presupuestos JSON → chunks → embeddings
│       ├── schemas.py          # Budget, BudgetComponent, Chunk, EmbeddedChunk, IngestRequest/Response
│       ├── chunker.py          # JSONStructuralChunker: un componente = un chunk, con cabecera del padre
│       ├── embedder.py         # OpenAIEmbedder: text-embedding-3-small por lotes, reintentos y coste
│       ├── similarity.py       # coseno, producto escalar y distancia euclidiana (biblioteca estándar)
│       ├── router.py           # POST /embeddings/ingest
│       └── SANITY_CHECK.md     # el deliverable del paso 7: las tres parejas y su lectura
├── prompts/                    # los prompts, fuera del código y versionados (ver prompts/README.md)
│   └── estimacion/
│       ├── v1/                 # template.toml + system.md.j2 + user.md.j2 (el prompt original, migrado)
│       ├── v2/                 # parámetros tipados en el mensaje de usuario, system prompt estable
│       ├── v3/                 # v2 + conversación con memoria: bloque <project_metadata> al final
│       └── v4/                 # v3 + compresión: bloques <anchors> y <summary> al final (por defecto)
│   └── resumen/v1/             # el prompt del summarizer (resumen acumulativo)
├── evals/
│   ├── metrics.py              # MetricResult, run_all_metrics y las métricas SchemaAdherence, CostBounds, ContentRecall
│   ├── golden_dataset.json     # 16 casos golden (transcripción + parámetros + lo que debe cumplir la estimación)
│   ├── run.py                  # runner de los casos golden (--http o in-process, --fake sin API key)
│   └── stress/                 # pre-work sesión 06: el stress del CAG
│       ├── scenarios.py        # tres perfiles multi-turno (growing, pivot, contradiction) con fact-tracker
│       ├── metrics.py          # LatencyBudgetMetric, CostBudgetMetric, MemoryDriftMetric
│       ├── fixtures/build_pdfs.py  # PDFs sintéticos calibrados (5/20/50/100 KB) con marcadores
│       ├── run.py              # runner: escenarios × tamaños × repeticiones → results.csv (+ REPORT.md)
│       ├── report.py           # REPORT.md a partir del CSV: tabla resumen, tres curvas, lectura
│       ├── results.csv         # una fila por turno con los 13 campos de turn_observed y las métricas
│       └── REPORT.md           # el deliverable de la sesión 06
├── app/db/                     # persistencia (sesión 08)
│   ├── models.py               # Base, Document, Chunk (vector(1536), metadata JSONB, cascada)
│   └── session.py              # motor async (asyncpg) y la dependencia de sesión por petición
├── streamlit_app.py            # conversación por turnos: formulario + adjuntos, panel de memoria, compresión e historial
├── .streamlit/                 # config.toml (headless) y secrets.toml.example (alternativa a .env)
├── tests/                      # pytest: API, sesiones (httpx.AsyncClient), adjuntos, extractor, plantillas, streaming e interfaz
├── scripts/
│   ├── estimar.sh              # envía una transcripción (con o sin parámetros) al endpoint con curl
│   ├── estimar-sesion.sh       # conversación de tres turnos con curl (el tercero con PDF y Word adjuntos)
│   ├── generar-adjuntos.py     # genera transcripciones/adjuntos/*.pdf y *.docx de ejemplo
│   ├── comparar.py             # compara transcripciones: tokens, coste, ahorro y estimación
│   ├── ingest_corpus.py        # ingesta los 15 presupuestos del corpus, uno por documento (sesión 08)
│   ├── generar-json.sh         # crea transcripciones/*.json a partir de los .txt
│   └── verificar.sh            # estructura + tests + arranque real + sesiones (lo usa el CI)
├── query_examples.py           # las cinco consultas semánticas del enunciado contra POST /search (sesión 08)
├── output_examples.txt         # su output real contra el corpus de 73 chunks
├── alembic.ini                 # config de Alembic; la URL sale de DATABASE_URL, no está escrita aquí
├── alembic/
│   ├── env.py                  # plantilla async + registro del tipo `vector` en el dialecto
│   └── versions/
│       └── 0001_initial_schema.py   # extensión vector + documents + chunks + índices no vectoriales
├── data/
│   └── budgets_sample.json     # 15 presupuestos históricos normalizados (sesión 07): 73 componentes, 4+ sectores
├── transcripciones/            # transcripciones de ejemplo (.txt, .json para curl, panaderia-turno-2/3 para la conversación)
│   └── adjuntos/               # especificación técnica (PDF) y propuesta anterior (Word) de la panadería
├── examples/                   # material del diagnóstico de la sesión 09 (ver ../arquitectura-actual.md)
│   ├── transcripts/            # 01_clear, 02_ambiguous y 03_hard: reuniones de cliente clara / ambigua / difícil
│   ├── trace_02_ambiguous.py   # script de cliente del trace: embebe, busca, ablaciona señal vs. ruido
│   └── trace_02_ambiguous.out.txt  # su output real contra el corpus de 73 chunks
├── docs/                       # comparativa de resultados (transcripción pobre vs. detallada)
├── Dockerfile                  # imagen basada en uv + Python 3.11 (copia app/, prompts/ y streamlit_app.py)
├── docker-compose.yml          # arranque con Docker, lee .env sin copiarlo a la imagen
├── .dockerignore
├── .env.example                # variables necesarias (sin valores)
├── .env                        # tus valores reales (ignorado por git y por docker)
├── pyproject.toml / uv.lock
└── README.md
```

## Requisitos

- [uv](https://docs.astral.sh/uv/) (instala solo el Python 3.11 fijado en `.python-version`), o bien
  [Docker Desktop](https://www.docker.com/get-started) para ejecutarlo todo en contenedores.
- Una API key de [OpenAI](https://platform.openai.com/) o de [Anthropic](https://console.anthropic.com/).
- **Docker**, también si se desarrolla con `uv`: desde la sesión 08 el pipeline vectorial necesita
  PostgreSQL con pgvector, y se levanta con `docker compose up postgres -d`. El estimador y las sesiones
  conversacionales (sesiones 02-06) siguen funcionando sin base de datos.

## Puesta en marcha

```bash
cd estimador-cag
uv sync                                  # crea .venv e instala dependencias
cp .env.example .env                     # rellena OPENAI_API_KEY o ANTHROPIC_API_KEY
uv run uvicorn app.main:app --reload     # http://localhost:8000
```

- Swagger UI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

Para el pipeline vectorial (`POST /embeddings/ingest` y `POST /search`) hacen falta además la base de datos y
sus migraciones:

```bash
docker compose up postgres -d            # PostgreSQL 16 + pgvector en localhost:5432
uv run alembic upgrade head              # crea documents y chunks
```

Interfaz web (ver "Interfaz: conversación por turnos"). Necesita la API arrancada, porque las sesiones
viven en el proceso de la API (`API_BASE_URL`, por defecto `http://localhost:8000`):

```bash
uv run streamlit run streamlit_app.py    # http://localhost:8501
```

Conversación de tres turnos por curl (crea sesión, dos turnos de texto y un tercero con PDF y Word adjuntos):

```bash
scripts/estimar-sesion.sh                # genera los adjuntos de ejemplo si no existen
```

## Ejecutar con Docker

```bash
cp .env.example .env                 # rellena la API key (solo la primera vez)
docker compose up --build -d         # API en http://localhost:8000 y formulario en http://localhost:8501
docker compose logs -f api chat      # ver los logs
docker compose down                  # parar
```

Levanta tres servicios: `postgres` (PostgreSQL 16 + pgvector, puerto 5432) y, con la misma imagen, `api`
(FastAPI, puerto 8000) y `chat` (la interfaz de Streamlit, puerto 8501, que llama a la API como
`http://api:8000` dentro de la red de compose). `api` no arranca hasta que `pg_isready` da por sana la base.

El esquema se crea aparte, con una migración (no se aplica en el arranque: una migración es un cambio
deliberado, no un efecto secundario de levantar el servicio):

```bash
docker compose run --rm api alembic upgrade head
```

Lanzar estimaciones por curl contra el contenedor:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/reunion-landing-page.json

curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/reunion-landing-page-parametros.json      # con parámetros tipados
```

Para desarrollar sobre el contenedor, `docker compose watch` sincroniza `app/` y `prompts/` y reinicia el
servicio al guardar; si cambian `pyproject.toml` o `uv.lock`, reconstruye la imagen.

Cómo está montado:

- La imagen parte de `ghcr.io/astral-sh/uv:python3.11-bookworm-slim` e instala las dependencias con
  `uv sync --frozen --no-dev` en una capa aparte, así solo se reinstalan si cambia `uv.lock`.
- El `.env` **no se copia a la imagen** (está en `.dockerignore`): `docker compose` lo inyecta como
  variables de entorno al arrancar. Sin `.env` el servicio arranca igualmente, pero el endpoint de
  estimación responde `500` hasta que se configure la API key.
- Los contenedores corren con un usuario sin privilegios y tienen `HEALTHCHECK` (`/health` en la API,
  `/_stcore/health` en Streamlit).

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `LLM_PROVIDER` | Proveedor a usar: `openai` o `anthropic` | `openai` |
| `OPENAI_API_KEY` | API key de OpenAI (si `LLM_PROVIDER=openai`) | — |
| `ANTHROPIC_API_KEY` | API key de Anthropic (si `LLM_PROVIDER=anthropic`) | — |
| `OPENAI_MODEL` | Modelo de OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Modelo de Anthropic | `claude-haiku-4-5` |
| `PROMPT_VERSION` | Versión de la plantilla de prompt por defecto (`prompts/estimacion/<versión>`) | `v4` |
| `SESSION_MAX_TURNS` | Turnos (par user+assistant) que conserva la ventana deslizante de cada sesión | `6` |
| `SESSION_MAX_SESSIONS` | Tope de sesiones vivas en memoria del proceso (se expulsa la menos reciente) | `500` |
| `ATTACHMENT_MAX_FILES` | Adjuntos máximos por petición | `5` |
| `ATTACHMENT_MAX_BYTES` | Tamaño máximo de cada adjunto | `10485760` (10 MB) |
| `ATTACHMENT_MAX_CHARS` | Caracteres de texto extraído que se conservan por adjunto (el resto se recorta y se avisa). 60.000 es el tope de referencia del curso; el pre-work 05 usaba 20.000 | `60000` |
| `ANCHORS_MAX` / `ANCHORS_PER_TURN` | Anclas que conserva una sesión (0 las desactiva) y tope por turno | `24` / `6` |
| `SUMMARY_MODE` | Resumen acumulativo de los turnos que salen de la ventana: `llm` (una llamada extra), `heuristic` (sin llamadas) u `off` | `llm` |
| `SUMMARY_MAX_CHARS` / `SUMMARY_PROMPT_VERSION` | Tope del resumen y versión de `prompts/resumen/` | `1500` / `v1` |
| `TIER_EXTENDED_MIN_CONTEXT_CHARS` / `TIER_EXTENDED_MIN_ATTACHMENT_CHARS` | Umbrales a partir de los que el turno resuelve el tier `extended` | `40000` / `15000` |
| `TIER_EXTENDED_MODEL` / `TIER_EXTENDED_MAX_TOKENS` | Modelo y tokens de salida del tier `extended` (por defecto, los mismos que el estándar: el tier solo se observa) | — |
| `RESPONSE_CACHE_ENABLED` / `SEMANTIC_CACHE_THRESHOLD` | Caché de respuestas y umbral de similitud (Jaccard) del acierto semántico | `true` / `0.9` |
| `LOG_FORMAT` | Formato de los eventos structlog (`turn_observed`): `json` o `console` | `json` |
| `API_BASE_URL` | URL de la API para la interfaz de Streamlit | `http://localhost:8000` |
| `LLM_TEMPERATURE` | Temperatura del modelo (0-2) | `0.2` |
| `LLM_MAX_TOKENS` | Máximo de tokens de salida | `4096` |
| `LLM_TIMEOUT_SECONDS` | Timeout de la llamada al proveedor | `60` |
| `LLM_INPUT_PRICE_PER_MTOK` | Precio en USD por millón de tokens de entrada (opcional, sobrescribe la tabla) | tabla de `pricing.py` |
| `LLM_OUTPUT_PRICE_PER_MTOK` | Precio en USD por millón de tokens de salida (opcional, sobrescribe la tabla) | tabla de `pricing.py` |

Las API keys solo viven en `.env`, que está en `.gitignore`. Nunca aparecen en el código.

Pre-work de la sesión 07 (pipeline de embeddings):

| Variable | Por defecto | Para qué |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings (siempre de OpenAI, usa `OPENAI_API_KEY` aunque `LLM_PROVIDER` sea `anthropic`). |
| `EMBEDDING_BATCH_SIZE` | `100` | Chunks por llamada a la API de embeddings. |

Pre-work de la sesión 08 (PostgreSQL + pgvector):

| Variable | Por defecto | Para qué |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://estimator:estimator@localhost:5432/estimator` | Conexión a PostgreSQL con el driver async. La usan la API y Alembic (`alembic.ini` no lleva la URL escrita). Dentro de compose, el host es `postgres`. |

## Probar el endpoint

Con el script (usa `transcripciones/reunion-landing-page.json` por defecto; acepta `.json` o `.txt`):

```bash
scripts/estimar.sh
scripts/estimar.sh transcripciones/reunion-landing-page-parametros.json   # con todos los parámetros
scripts/estimar.sh transcripciones/reunion-app-reservas-restaurante.json
```

Con curl y parámetros tipados (todos opcionales salvo `transcription`; un campo desconocido o un valor
fuera de rango devuelve `422`):

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG. El diseño ya existe en Figma.",
    "project_name": "Landing page con HubSpot",
    "project_type": "web",
    "hourly_rate_eur": 60,
    "deadline_weeks": 4,
    "estimation_mode": "equilibrada",
    "detail_level": "resumen",
    "prompt_version": "v2"
  }'
```

| Campo | Tipo | Valores | Por defecto |
|---|---|---|---|
| `transcription` | texto, mínimo 20 caracteres | — | obligatorio |
| `project_name` | texto, máximo 100 | — | el modelo lo deduce |
| `project_type` | enumerado | `web`, `movil`, `api`, `datos`, `otro` | el modelo lo deduce |
| `hourly_rate_eur` | número, 0 < x ≤ 1000 | — | `50` |
| `deadline_weeks` | entero, 1-104 | — | sin plazo |
| `estimation_mode` | enumerado | `conservadora`, `equilibrada`, `optimista` | `conservadora` |
| `detail_level` | enumerado | `completa`, `resumen` | `completa` |
| `prompt_version` | `v<N>` | las de `GET /api/v1/prompts` | `PROMPT_VERSION` |

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
  "usage": {
    "input_tokens": 1954,
    "output_tokens": 331,
    "total_tokens": 2285,
    "cost": {
      "input_usd": 0.000293,
      "output_usd": 0.000199,
      "total_usd": 0.000492,
      "input_price_per_mtok": 0.15,
      "output_price_per_mtok": 0.6,
      "currency": "USD"
    }
  },
  "prompt": {
    "template": "estimacion",
    "version": "v2",
    "system_prompt_chars": 10414,
    "user_prompt_chars": 1260,
    "transcription_chars": 280
  },
  "generated_at": "2026-09-28T17:30:10.585676Z"
}
```

- `usage.input_tokens` son los tokens del system prompt (contexto CAG) más los parámetros y la
  transcripción; `usage.output_tokens`, los de la estimación generada.
- `usage.cost` es el coste de la llamada según el precio de lista del modelo (ver "Tokens y coste").
- `prompt` dice qué plantilla y versión generaron la estimación y cuánto ocupa cada parte: el contexto
  fijo (`system_prompt_chars`, igual en todas las llamadas), la parte variable (`user_prompt_chars`:
  parámetros + transcripción) y la transcripción sola.

`GET /api/v1/prompts` lista las versiones de plantilla disponibles, con su descripción, fecha y changelog,
y cuál es la de por defecto.

Códigos de respuesta: `200` estimación generada · `422` body inválido o versión de plantilla inexistente ·
`500` falta la API key · `502` el proveedor ha fallado.

## Sesiones conversacionales con memoria y adjuntos (pre-work sesión 05)

Hasta la sesión 04 el estimador era transaccional: una transcripción entra, una estimación sale. En una
empresa real el cliente refina el alcance en varias reuniones, manda documentos y espera que el sistema
recuerde de qué proyecto se habla. El pre-work de la sesión 05 añade eso con dos piezas separadas a
conciencia:

| | Qué es | Dónde vive | Cómo viaja al LLM |
|---|---|---|---|
| **Historial** | Los mensajes `user`/`assistant` de los turnos anteriores, en bruto | `ConversationHistory` (`app/services/sessions.py`) | En el array `messages`, recortado por una **ventana deslizante** de `SESSION_MAX_TURNS` turnos (por defecto 6; un turno es un par user+assistant) |
| **Memoria** | Los hechos del proyecto en curso: nombre, tipo, alcance acordado, tecnologías, equipo asumido, última estimación, tarifa, plazo, documentos | `ProjectMetadata` (Pydantic, mismo módulo) | En el bloque `<project_metadata>` **al final del system prompt** (plantilla `v3`), regenerado en cada turno |

La diferencia operativa: el historial es texto que crece y se descarta (cuando el turno 1 sale de la
ventana, el LLM ya no lo ve); la memoria es conocimiento estructurado y pequeño que sobrevive a la ventana
(el nombre del proyecto se fijó en el turno 1 y sigue en el system prompt en el turno 8). Las dos viven en
una `Session`, y las sesiones en `SessionStore`: **un diccionario en memoria del proceso**, sin base de
datos ni Redis. Se acepta esa volatilidad porque el objetivo del ejercicio es separar historial de memoria
y gobernar lo que viaja al LLM; una sesión dura minutos; y la interfaz del store (`create`/`get`/`delete`)
es la misma que tendría uno respaldado en Redis o Postgres cuando haya varios workers o haga falta retomar
sesiones (los docstrings del módulo lo detallan).

### Endpoints

```bash
# 1) Crear la sesión (201). El session_id (UUID v4) viaja en cada petición posterior.
curl -s -X POST http://localhost:8000/api/v1/sessions
# {"session_id":"bae1820a-...","created_at":"2026-10-05T23:07:27Z","max_turns":6}

# 2) Turno 1: multipart/form-data con la transcripción y los parámetros tipados del formulario
curl -s -X POST http://localhost:8000/api/v1/sessions/$SID/estimate \
  -F "transcript=<transcripciones/panaderia-descripcion-detallada.txt" \
  -F "project_name=Tienda online La Espiga" -F "project_type=web" \
  -F "hourly_rate_eur=50" -F "deadline_weeks=8" -F "detail_level=resumen"

# 3) Turno 2: solo lo nuevo. Sin nombre de proyecto: lo recuerda la memoria
curl -s -X POST http://localhost:8000/api/v1/sessions/$SID/estimate \
  -F "transcript=<transcripciones/panaderia-turno-2-cambios.txt" -F "detail_level=resumen"

# 4) Turno 3: con adjuntos (PDF y Word); se repite -F attachments por cada fichero
curl -s -X POST http://localhost:8000/api/v1/sessions/$SID/estimate \
  -F "transcript=<transcripciones/panaderia-turno-3-adjunto.txt" \
  -F "attachments=@transcripciones/adjuntos/panaderia-especificacion-tecnica.pdf;type=application/pdf" \
  -F "attachments=@transcripciones/adjuntos/panaderia-propuesta-anterior.docx"

# Estado de la sesión: memoria del proyecto e historial efectivo (lo que viajaría al LLM)
curl -s http://localhost:8000/api/v1/sessions/$SID
curl -s -X DELETE http://localhost:8000/api/v1/sessions/$SID      # 204
```

Campos del multipart de `POST /api/v1/sessions/{session_id}/estimate`:

| Campo | Tipo | Notas |
|---|---|---|
| `transcript` | texto, mínimo 20 caracteres | **obligatorio**. Lo nuevo de este turno (no hace falta repetir lo anterior) |
| `attachments` | ficheros, 0..`ATTACHMENT_MAX_FILES` | `.pdf`, `.docx`, `.txt`, `.md`. Un `.doc` o una imagen devuelven `415` |
| `project_name`, `project_type`, `hourly_rate_eur`, `deadline_weeks`, `estimation_mode`, `detail_level`, `prompt_version` | como en `POST /api/v1/estimate` | opcionales; se validan con el mismo `EstimationRequest` (`422` con el mismo formato de error) |

La respuesta es la de `POST /api/v1/estimate` (`estimation`, `model`, `provider`, `usage`, `prompt`,
`generated_at`) más el estado de la sesión:

```json
{
  "estimation": "## Estimación: Tienda online La Espiga\n...",
  "usage": { "input_tokens": 6291, "output_tokens": 419, "total_tokens": 6710, "cost": { "total_usd": 0.001195, ... } },
  "prompt": { "template": "estimacion", "version": "v3", "system_prompt_chars": 13044, "user_prompt_chars": 3694, "transcription_chars": 389 },
  "session_id": "bae1820a-...",
  "turn": 3,
  "window": { "max_turns": 6, "turns_in_window": 3, "dropped_turns": 0, "messages_sent": 6 },
  "project_metadata": {
    "project_name": "Tienda online La Espiga", "project_type": "web",
    "assumed_team": "1 desarrollador backend + 1 desarrollador frontend + 1 diseñador UX (part-time)", "assumed_team_size": 3,
    "mentioned_technologies": ["WhatsApp", "Bizum", "Stripe", "Excel", "AWS", "WordPress", "WooCommerce"],
    "agreed_scope": "Desarrollo de una tienda online para la panadería ...", "hourly_rate_eur": 50, "deadline_weeks": 8,
    "last_total_hours": 324, "attached_documents": ["panaderia-especificacion-tecnica.pdf", "panaderia-propuesta-anterior.docx"],
    "updated_at": "2026-10-05T23:08:12Z"
  },
  "attachments": [
    { "filename": "panaderia-especificacion-tecnica.pdf", "kind": "pdf", "size_bytes": 2371, "chars": 1290, "pages": 2, "truncated": false, "warning": null },
    { "filename": "panaderia-propuesta-anterior.docx", "kind": "docx", "size_bytes": 37156, "chars": 683, "pages": null, "truncated": false, "warning": null }
  ]
}
```

Códigos: `201` sesión creada · `200` turno completado · `404` la sesión no existe (o el servicio se
reinició) · `413` adjunto demasiado grande · `415` formato de adjunto no soportado · `422` formulario
inválido, versión de plantilla inexistente, adjunto ilegible o demasiados adjuntos · `500`/`502` como en
`/estimate`. Si el LLM falla, el turno se retira del historial: un error no deja un mensaje de usuario sin
respuesta.

### La ventana deslizante

`ConversationHistory` guarda turnos (`user` + `assistant`). `start_turn(user)` mete el mensaje del turno
actual y recorta: con `max_turns=6` viajan como mucho 6 mensajes de usuario (los 5 anteriores con su
respuesta más el actual) además del system prompt. `to_messages_list(system_prompt)` devuelve el array
listo para la API con el system prompt **regenerado fuera** a partir de la memoria actual: el system prompt
no se guarda en el historial, así nunca sale de la ventana y siempre lleva la memoria más reciente.

Por qué es la estrategia por defecto razonable al arrancar: es trivial, determinista, barata (ninguna
llamada extra) y acota el coste por turno. Qué la hace insuficiente, y se ve en la conversación real de
abajo: la entrada crece un turno tras otro (3.894 → 4.917 → 6.291 tokens en tres turnos) porque cada
mensaje de usuario repite parámetros y recordatorios y el adjunto se queda dentro del historial; y cuando un
turno sale de la ventana se pierde íntegro aunque contuviera la única mención a un requisito. La memoria
amortigua lo segundo (los hechos extraídos sobreviven), pero no lo primero. Eso es lo que resolverán en el
directo la compresión con anclas y el resumen acumulativo.

### Adjuntos: camino B (extracción local)

De los dos caminos del enunciado se implementa el **B**: `app/services/attachments.py` extrae el texto con
`pypdf` (PDF) y `python-docx` (Word; párrafos y tablas en el orden del documento), acepta también `.txt` y
`.md`, y lo concatena a la transcripción con separadores claros antes de renderizar el prompt:

```
<transcripción del turno>

--- attachment: panaderia-especificacion-tecnica.pdf ---
<texto extraído>
--- fin del adjunto: panaderia-especificacion-tecnica.pdf ---
```

Por qué B y no A (subir el fichero a la Files API del proveedor):

- **Independiente del proveedor.** El proyecto soporta OpenAI y Anthropic con el mismo código; con el
  camino A habría que mantener dos integraciones de ficheros y el adjunto solo funcionaría con modelos
  multimodales.
- **Control y medida.** El texto que ve el modelo es inspeccionable (`attachments[].chars`, truncado o no,
  páginas), se puede recortar (`ATTACHMENT_MAX_CHARS`) y cuenta en `user_prompt_chars` y en los tokens. Un
  PDF escaneado sin texto no pasa en silencio: llega con `warning`.
- **Se prueba sin LLM.** Los tests generan PDF y Word reales y comprueban la extracción y que el texto
  llega al prompt; con la Files API habría que llamar al proveedor.
- **Prepara RAG.** Convertir el documento en texto es el primer paso del chunking del módulo 3; con el
  camino A ese paso no existe.

Lo que se pierde: diagramas y tablas complejas (el PDF se reduce a texto), y hay que mantener la
extracción. Es la contrapartida aceptada.

### Cómo se actualiza `project_metadata`: heurística, no LLM extractor

Tras cada respuesta, `app/services/metadata_extractor.py` extrae los hechos del turno **sin ninguna
llamada adicional** y `ProjectMetadata.merged` los fusiona con la memoria anterior (el dato nuevo manda,
un dato ausente no borra, las listas se acumulan sin duplicados):

| Campo | De dónde sale |
|---|---|
| `project_name` | El parámetro del formulario; si no, el título `## Estimación: ...` de la **primera** estimación. Después solo lo cambia el formulario (ver observaciones) |
| `project_type`, `hourly_rate_eur`, `deadline_weeks` | Los parámetros del turno |
| `assumed_team`, `assumed_team_size` | La línea `**Equipo recomendado: ...**`; el tamaño suma las cifras ("1 backend + 1 frontend + 1 diseñador" → 3) o cuenta perfiles |
| `agreed_scope` | La sección `### Resumen del alcance` de la última estimación: es el propio modelo consolidando lo acordado |
| `last_total_hours` | `**Total estimado: N horas**` (admite "1.250" y "47,5") |
| `mentioned_technologies` | Un vocabulario cerrado de ~100 tecnologías buscado en transcripción, adjuntos y estimación (límites de palabra; "React Native" no cuenta también como "React") |
| `attached_documents` | Los nombres de los adjuntos de la sesión |

Por qué heurística: el prompt `v2`/`v3` ya obliga al modelo a un **formato de salida fijo** con títulos y
líneas conocidas. Ese formato es un contrato que ya se paga en el prompt; parsearlo es barato, determinista
y se prueba con pytest sin LLM. Un extractor LLM duplicaría llamadas, latencia y coste por turno y añadiría
otra salida que validar. Su límite es evidente (el vocabulario de tecnologías es cerrado y el alcance es el
resumen literal del modelo), y por eso `extract_project_metadata` es el único punto a sustituir si más
adelante compensa el extractor LLM.

### La plantilla `v3`: la memoria va al final del system prompt

`prompts/estimacion/v3` es `v2` más una sección "Conversación con memoria" (cada respuesta es la
estimación completa y actualizada del proyecto; el dato más reciente manda y los cambios se dejan constar en
supuestos; cómo leer los adjuntos delimitados) y el bloque `<project_metadata>`, que va **después de los
ejemplos CAG**, al final. Vacío en el primer turno:

```
<project_metadata>
</project_metadata>
```

y con los hechos conocidos a partir del segundo:

```
<project_metadata>
- Nombre del proyecto: Tienda online La Espiga
- Tipo de proyecto: web
- Alcance acordado hasta ahora: Desarrollo de una tienda online para la panadería ...
- Tecnologías mencionadas: WhatsApp, Bizum, Stripe, Excel
- Equipo asumido: 1 desarrollador backend + 1 desarrollador frontend + 1 diseñador UX (part-time) (3 personas)
- Última estimación: 288 horas
- Última tarifa: 50 €/hora
- Último plazo objetivo: 8 semanas
</project_metadata>
```

Que vaya al final no es estética: el prefijo del system prompt (rol, reglas, formato, ejemplos: ~12.000
caracteres) sigue siendo **idéntico entre turnos y entre sesiones**, así que sigue siendo cacheable en el
proveedor; solo cambia la cola. Los tests lo comprueban (`test_v3_inyecta_la_memoria_al_final_sin_tocar_el_prefijo`).
`v1` y `v2` siguen funcionando en una sesión (el historial viaja igual), pero ignoran la memoria: sirve para
comparar.

### Conversación real de tres turnos (`scripts/estimar-sesion.sh`, gpt-4o-mini, 2026-10-05)

Transcripción detallada de la panadería → cambios de alcance una semana después → especificación técnica
(PDF, 2 páginas) y propuesta anterior de otra consultora (Word) adjuntas:

| Turno | Mensajes al LLM | Tokens entrada | Tokens salida | Coste (USD) | Total estimado | Memoria tras el turno |
|---|---:|---:|---:|---:|---:|---|
| 1. Transcripción detallada, con nombre y parámetros | 2 | 3.894 | 378 | 0.000811 | 288 h | nombre, tipo, alcance, equipo (3), tecnologías WhatsApp, Bizum, Stripe, Excel; 288 h |
| 2. Cambios: reparto a 3 km, suscripción semanal, Bizum (sin repetir nombre ni transcripción) | 4 | 4.917 | 419 | 0.000989 | 312 h | mismo nombre; alcance actualizado con los pedidos recurrentes; 312 h |
| 3. "Revisad con estos documentos" + PDF (1.290 caracteres) + Word (683) | 6 | 6.291 | 419 | 0.001195 | 324 h | + AWS, WordPress, WooCommerce; documentos registrados; 324 h |

Qué se ve:

- **Coherencia entre turnos.** Las tres estimaciones se titulan `Tienda online La Espiga` y el turno 2
  integra los cambios sobre la estimación anterior en vez de empezar de cero (el total sube de 288 a 312 h
  por la suscripción y el cálculo del radio). El turno 3 incorpora la especificación (AWS, bilingüe,
  exportación a Excel) y sube a 324 h.
- **El mensaje de usuario del turno 2 es corto** (2.040 caracteres frente a 3.660 del primero) y aun así el
  modelo sabe de qué proyecto se habla: lo sabe por el historial (4 mensajes) y por la memoria del system prompt.
- **La entrada crece** 3.894 → 4.917 → 6.291 tokens: es el historial acumulándose. Con 6 turnos y adjuntos
  la ventana por sí sola no basta; es el problema que atacará el directo.
- **El coste total de la conversación** fue de 0.003 USD.

Dos cosas salieron solo probando, y están corregidas en la rama:

1. **El nombre del proyecto derivaba.** En la primera ejecución, el turno 2 sin `project_name` tituló la
   estimación "Tienda online Forn de la Plaça" y el turno 3 "Tienda online Panadería Artesana La Espiga",
   aunque la memoria decía "Tienda online La Espiga". Causa: el mensaje de usuario de `v2` decía "no indicado
   (dedúcelo de la transcripción)" y esa instrucción, más cercana, ganaba a la memoria. Arreglo doble: `v3`
   pide usar el nombre de `<project_metadata>` antes de deducir uno, y el extractor solo toma el título de
   la estimación cuando la memoria aún no tiene nombre (después solo lo cambia el formulario). Lección: la
   memoria inyectada compite con las instrucciones del mensaje de usuario; hay que alinearlas.
2. **El formato `resumen` recorta.** En esa misma ejecución el turno 3 bajó a 246 h: con "máximo 5 bloques"
   el modelo dejó fuera testing y despliegue. No es un problema de memoria, sino del formato: para
   conversaciones largas conviene `completa`, o un `resumen` cuyos bloques transversales sean obligatorios.

### Límites y decisiones

- Las sesiones **se pierden al reiniciar** el servicio y no se comparten entre workers (un diccionario por
  proceso). `SESSION_MAX_SESSIONS` acota la memoria expulsando la sesión menos reciente. La interfaz detecta
  el `404` y crea una sesión nueva.
- Un turno a la vez por sesión (`asyncio.Lock`): dos peticiones concurrentes al mismo `session_id` no
  intercalan sus mensajes.
- El endpoint de sesión **no hace streaming**: para actualizar historial y memoria hay que tener la respuesta
  completa, y la interfaz pasa a hablar con la API por HTTP. Se perdió el `st.write_stream` de la sesión 04 a
  cambio de que el estado viva en la API, que es donde tiene sentido. Un endpoint de streaming con SSE que
  cierre el turno al terminar es trabajo para más adelante.
- Los adjuntos quedan **dentro del historial** (en el mensaje de usuario del turno en que se enviaron) hasta
  que ese turno sale de la ventana; mientras tanto se pagan en cada llamada. La memoria conserva sus
  tecnologías y su nombre, no su contenido. Otro argumento para la compresión del directo.

## Compresión del historial, tier, caché y observación (sesión 05 directo)

El pre-work de la 05 dejó dos problemas medidos: la entrada crece turno a turno y, cuando un turno sale de
la ventana, se pierde entero aunque fuera el único que mencionaba un requisito. Esto es lo que lo resuelve
(la plantilla `v4` es `v3` más dos bloques al final del system prompt; el prefijo cacheable no cambia):

| Pieza | Módulo | Qué hace | Cómo viaja al LLM |
|---|---|---|---|
| **Anclas** | `app/services/anchors.py` | De cada mensaje del usuario extrae hechos cortos con reglas deterministas: nombre (`name`), presupuesto (`budget`), plazo (`deadline`), tecnologías (`stack`), frases de cambio (`change`) y requisitos (`requirement`). Los de un solo valor (nombre, presupuesto, plazo) se sustituyen por el más reciente; el resto se acumula sin duplicados hasta `ANCHORS_MAX` (24), expulsando los acumulables más antiguos | Bloque `<anchors>`: `- [turno k] Presupuesto: 30.000 €` |
| **Resumen acumulativo** | `app/services/summarizer.py` | Cuando `start_turn` expulsa un turno de la ventana, **antes** de llamar al LLM ese turno se pliega en `Session.summary`. `SUMMARY_MODE=llm` hace una llamada extra con `prompts/resumen/v1` (se mide y se paga: `summarizer_*` en `turn_observed`); `heuristic` guarda el principio de lo que dijo el cliente y las líneas clave de la estimación sin llamar a nadie; `off` es el comportamiento anterior. Si la llamada falla se degrada a heurístico y el turno no se pierde. Tope `SUMMARY_MAX_CHARS` (1.500): cuando no cabe, se pierde lo más antiguo | Bloque `<summary>` |
| **Tier dinámico** | `app/services/tiering.py` | Reglas en orden sobre el contexto del turno: `attachments_large` (adjuntos ≥ 15.000 caracteres) y `context_large` (system + ventana ≥ 40.000) resuelven `extended`; si no, `standard` (`default`). Cada turno registra `last_resolved_tier` y `last_tier_rule`. Por defecto los dos tiers usan **el mismo modelo**: el tier se observa sin alterar las medidas (`TIER_EXTENDED_MODEL` lo cambia) | Elige `model` y `max_tokens` de la llamada |
| **Caché de respuestas** | `app/services/response_cache.py` | Exacta (hash del system prompt y todos los mensajes) y semántica (mismo system e historial, y el mensaje actual con Jaccard de palabras ≥ `SEMANTIC_CACHE_THRESHOLD`, 0.9). Sin embeddings, a propósito: determinista y sin dependencias. Un acierto devuelve la estimación guardada con 0 tokens y 0 USD y `cache_hit_kind` ∈ {`exact`, `semantic`} | Evita la llamada |
| **`turn_observed`** | `app/services/observability.py` | Un único evento structlog por turno con los 13 campos del ejercicio de la sesión 06 (`turn_index`, `session_id`, `enriched_transcript_chars`, `attachments_total_chars`, `messages_in_window`, `anchors_count`, `summary_chars`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`, `cache_hit_kind`, `last_resolved_tier`) más extras (regla del tier, modelo, `summarizer_*`, `context_chars`...). `tokens_in`, `cost_usd` y `latency_ms` incluyen al summarizer. Se emite en JSON a stdout y se guarda en la sesión | — |

Por qué un evento agregado y no cinco logs sueltos: el CSV del stress se extrae de una pasada, la
correlación entre `messages_in_window` y `cost_usd` es trivial (misma fila) y no hay que reconciliar
timestamps. Y por qué además se expone por la API: `POST /api/v1/sessions/{id}/estimate` devuelve el evento
en `turn_observed` (y `cache_hit_kind`, `tier`, `compression`), y `GET /api/v1/sessions/{id}` devuelve
`message_count`, `anchors_count`, `summary_chars`, `last_resolved_tier`, `last_tier_rule`, el bloque
`compression` (anclas y resumen) y `last_turn_observed`. Así el runner funciona igual contra Docker, uvicorn o
in-process, sin parsear stdout. Es el mismo diccionario que va al log (`emit_turn_observed`).

```bash
# Un turno con todo: el evento en la respuesta...
curl -s -X POST http://localhost:8000/api/v1/sessions/$SID/estimate -F "transcript=<transcripciones/panaderia-turno-2-cambios.txt" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["cache_hit_kind"], d["tier"]); print(json.dumps(d["turn_observed"], indent=1, ensure_ascii=False))'
# ...y en el log del servicio (una línea JSON por turno)
docker compose logs api | grep turn_observed
```

El orden dentro del turno (`app/services/conversation.py`) importa: validar → `start_turn` (sale el turno
más antiguo) → summarizer sobre lo que salió → anclas del turno → system prompt con memoria + anclas +
resumen → tier → caché o LLM → cierre (historial, memoria, anclas, tier) → `turn_observed`. Si el LLM
falla, el turno y sus anclas se retiran; el resumen de lo que ya salió se conserva porque esos turnos ya no están.

El patrón Actor-Critic-Boss (opcional en la 05) **no está implementado**; `evals/run.py --mode acb` lo dice en
vez de fingirlo.

### Framework de evals (`evals/`)

`evals/metrics.py` define el patrón: `MetricResult(name, score, passed, details)`, un `Protocol`
`Metric.evaluate(observation)` y `run_all_metrics`, que nunca lanza (una métrica que explota cuenta como
fallida con el error en `details`). Tres métricas sobre una estimación suelta (`EvalObservation`):

| Métrica | Qué comprueba | Tipo de fallo que detecta |
|---|---|---|
| `SchemaAdherenceMetric` | Las secciones y líneas obligatorias del formato según `detail_level` (y ninguna prohibida), y que el total se parsea | **Duro**: si el formato se rompe, `metadata_extractor` deja de extraer |
| `CostBoundsMetric(max_usd)` | `cost_usd ≤` tope del caso (o global) | Presupuesto como contrato |
| `ContentRecallMetric(threshold)` | Fracción de los términos esperados presentes en la estimación | **Silencioso**: la salida es válida y barata, pero vacía |

`evals/golden_dataset.json` trae 16 casos (transcripción + parámetros + términos esperados + tope de coste) y
`evals/run.py` los ejecuta contra `POST /api/v1/estimate`:

```bash
uv run python -m evals.run --http http://localhost:8000          # o sin --http: in-process con tu .env
uv run python -m evals.run --cases landing-hubspot,api-pagos-stripe --output evals/results.json
uv run python -m evals.run --fake                                 # humo del runner con el LLM simulado (no mide calidad)
```

## Stress del CAG (pre-work sesión 06)

El deliverable es [`evals/stress/REPORT.md`](evals/stress/REPORT.md) con [`evals/stress/results.csv`](evals/stress/results.csv):
el baseline cuantitativo del CAG (latencia, coste, memoria) sobre el que el directo compara RAG. Nada se
optimiza entre medidas: ventana de 6, resumen de 1.500 caracteres, 24 anclas, tope de adjunto de 60.000.

| Bloque | Dónde | Qué |
|---|---|---|
| 1. Observación por turno | `app/services/observability.py` | El evento `turn_observed` de arriba |
| 2. Escenarios multi-turno | `evals/stress/scenarios.py` | Tres perfiles de 20 turnos sobre un mismo proyecto, cada turno con su **fact-tracker** (hechos literales que los turnos posteriores deberían recordar) y, en `pivot` y `contradiction`, los hechos que un turno **retira**: `growing` (Nimbus: un requisito nuevo por turno; ¿sobrevive el nombre al turno 20?), `pivot` (Atlas: React Native → Flutter en el turno 5; ¿`mentioned_technologies` acumula ambos?), `contradiction` (Faro: 30.000 € en el turno 3, 80.000 € en el 8, plazo de 12 a 20 semanas en el 12; ¿cuál se preserva, cuál es ancla, cuál acaba en el resumen?) |
| 3. Adjuntos grandes | `evals/stress/fixtures/build_pdfs.py` | PDFs sintéticos deterministas calibrados por caracteres extraíbles (5.000 / 20.000 / 50.000 / 100.000; 0 = sin adjunto) con cinco marcadores al 2 %, 25 %, 50 %, 75 % y 98 % del documento (Redsys, Power BI, Azure, SAP, catalán). El de 100 KB supera el tope de 60.000 y mide cómo trunca. No se versionan: el runner los regenera |
| 4. Métricas | `evals/stress/metrics.py` | `LatencyBudgetMetric(budget_ms)`, `CostBudgetMetric(budget_usd)` y `MemoryDriftMetric(fact, where=["summary","anchors","metadata"])`: match exacto case-insensitive sobre el snapshot de `GET /sessions/{id}`. Viven aquí y no en `evals/metrics.py` porque evalúan el shape de `turn_observed` y del snapshot de sesión, no una estimación suelta; comparten `MetricResult` y `run_all_metrics`. Tests en `tests/test_stress_metrics.py` (pasa, falla, límite por métrica) |
| 5. Runner + reporte | `evals/stress/run.py`, `evals/stress/report.py` | Escenarios × tamaños × repeticiones → una sesión por combinación, el adjunto en el turno 1 → una fila por turno en el CSV (los 13 campos + extras + una columna por métrica + recall del adjunto) → `REPORT.md` (tabla resumen, tres curvas en tabla, lectura) |

```bash
uv run python -m evals.stress.run --http http://localhost:8000 \
    --scenarios growing,pivot,contradiction \
    --attachment-sizes 0,5,20,50,100 \
    --repeats 3 \
    --output evals/stress/results.csv            # y REPORT.md al terminar (--report, --no-report)
uv run python -m evals.stress.report              # regenera REPORT.md desde el CSV
uv run python -m evals.stress.run --fake --turns 8 --repeats 2   # humo sin API key (el reporte avisa de que no es un baseline)
```

Flags útiles: `--turns N` acota los turnos por escenario, `--latency-budget-ms` (4.000) y
`--cost-budget-usd` (0.01) son los presupuestos que `LatencyBudgetMetric` y `CostBudgetMetric` convierten en
test. Sin `--http` corre in-process con el `.env` del proyecto. La tirada completa son 3 × 5 × 3 × 20 = 900
turnos, de los que las repeticiones 2 y 3 aciertan en la caché exacta (son el test de **ráfagas**: mismo
prompt, coste 0); la tirada real del 2026-10-12 con `gpt-4o-mini` hizo 718 llamadas reales, costó 1,53 USD y
tardó 106 minutos. El CSV se vuelca tras cada conversación: un fallo a mitad no pierde lo medido. **Arranca el
servicio justo antes de la tirada**: la caché de respuestas vive en memoria del proceso y, si el servicio ya
respondió a esos prompts (por ejemplo en una pasada de humo), la repetición 1 acierta en caché desde el primer
turno y deja de medir al LLM.

Qué mide cada curva del reporte:

- **Latencia vs tokens de entrada** (solo turnos sin caché): la pendiente del proveedor, con el summarizer dentro.
- **Coste acumulado vs turno** (sin adjunto, primera repetición): dónde se dobla la pendiente (turno 7: entra el summarizer).
- **MemoryDriftMetric vs N** (N ∈ {1, 3, 6, 10, 20}): la degradación silenciosa. No mira el historial: mide lo que
  sobrevive a la ventana. Un hecho reformulado por el summarizer cuenta como perdido, y eso se discute en el
  reporte, no se maquilla en la métrica (determinismo > sofisticación).
- **Adjuntos** (turno 1): tokens, latencia, coste, tier y recall de los marcadores del PDF en la estimación.

### Lo que enseñó la tirada real (gpt-4o-mini, 900 turnos, 1,53 USD)

Las cifras completas están en [`evals/stress/REPORT.md`](evals/stress/REPORT.md); esto es lo que cambia la
forma de ver el CAG:

- **El coste es la dimensión que más se degrada**: el turno 20 cuesta 3,3 veces el turno 1 (0,0024 frente a
  0,00075 USD) con 3,3 veces más tokens de entrada, aunque ningún turno supera el presupuesto de 0,01 USD. La
  pendiente se dobla en el turno 7, cuando entra el summarizer. Ventana y resumen ponen un techo: la entrada se
  estabiliza en 10.000-12.000 tokens en vez de crecer sin límite.
- **La latencia no la manda la entrada, sino las llamadas encadenadas y la salida**: el turno 1 con un adjunto
  de 50-100 KB (15.000-17.500 tokens de entrada, una sola llamada) responde en unos 4 s; los turnos 7-20
  (8.000-12.000 tokens) tardan 8-12 s porque suman la llamada del summarizer (1,5-3,5 s) y generan el doble
  de salida (900-1.500 tokens frente a 400-500: la estimación crece con el proyecto). El 94 % de los turnos
  sin caché supera el presupuesto de 4 s; solo los turnos 1-3 lo cumplen.
- **La caché exacta solo sirve mientras no entra el summarizer**: en las ráfagas (repeticiones 2 y 3) acierta
  el 100 % de los turnos 1-6 y el 0 % de los turnos 7-20, porque el resumen que escribe el LLM cambia entre
  pasadas y el system prompt deja de ser idéntico. La semántica no acertó nunca (exige el mismo system). Si se
  quiere caché con summarizer, hay que excluir el bloque `<summary>` de la clave o usar el modo heurístico.
- **La memoria literal no se degrada en 20 turnos**: recall del fact-tracker del 100 % en los tres escenarios
  al turno 20 y el nombre del proyecto en el título el 100 % de las veces. Anclas, resumen y `project_metadata`
  se solapan (la mayoría de los hechos vive en dos o tres sitios). Lo que falla es **retirar**: el 100 % de las
  conversaciones de `pivot` llega al turno 20 con React Native aún presente y las de `contradiction` con los
  30.000 € y las 12 semanas antiguas. Las anclas de un solo valor sí se sustituyen; el resumen (por diseño) y
  `mentioned_technologies` (por construcción) solo acumulan.
- **Adjuntos**: no cuestan latencia (el turno 1 responde en ~4 s con 0 o con 100 KB) pero sí recall. El de
  100 KB se trunca a 60.000 caracteres y solo el 40 % de los marcadores llega a la estimación; con los que caben
  enteros (5, 20 y 50 KB) el modelo ya deja fuera marcadores (recall medio 87 %). El tier `extended` se activa
  por adjunto desde 20 KB y por contexto desde el turno 14 en `growing`.
- **Los dos párrafos de lectura del reporte se generan a partir del CSV** (`evals/stress/report.py`): las
  afirmaciones cuantitativas (ratios, porcentajes, turno de ruptura) se recalculan al regenerar, así que no
  pueden quedarse desactualizadas respecto a los datos.

## Pipeline de embeddings (pre-work sesión 07)

Primer paso hacia la búsqueda semántica: convertir los presupuestos históricos (la salida normalizada de la
sesión 06) en vectores. Ni la interfaz ni el estimador se tocan.

> **El contrato de `POST /embeddings/ingest` que se describe aquí es el de la sesión 07** (recibía
> `{"budgets": [...]}` y devolvía los vectores por HTTP). La sesión 08 lo sustituyó por un documento por
> petición con persistencia en PostgreSQL: ver [Persistencia en PostgreSQL + pgvector](#persistencia-en-postgresql--pgvector-pre-work-sesión-08).
> Lo que sigue vigente sin cambios es el **chunker** (un componente = un chunk, con cabecera de contexto del
> presupuesto padre), el **embedder** por lotes y las medidas de tokens y coste.

```
presupuestos JSON → JSONStructuralChunker → chunks (texto + metadata + token_count)
                    (un componente = un chunk)        ↓
                                              OpenAIEmbedder (lotes de 100)
                                                      ↓
                                     EmbeddedChunk[] + stats (tokens, coste)
```

### Arrancar y usar el endpoint

```bash
# local
uv run uvicorn app.main:app --reload
# o con Docker
docker compose up --build -d

# los 15 presupuestos del sample
curl -s -X POST http://localhost:8000/embeddings/ingest \
  -H "Content-Type: application/json" \
  -d "{\"budgets\": $(cat data/budgets_sample.json)}" | python -m json.tool | head -40

# solo las estadísticas (los vectores son 73 × 1536 floats: mejor no volcarlos a la terminal)
curl -s -X POST http://localhost:8000/embeddings/ingest \
  -H "Content-Type: application/json" \
  -d "{\"budgets\": $(cat data/budgets_sample.json)}" \
  | python -c "import json,sys; print(json.load(sys.stdin)['stats'])"
```

También se puede invocar desde Swagger en <http://localhost:8000/docs> (sección **Embeddings**), pegando el
contenido de `data/budgets_sample.json` dentro de `{"budgets": ...}`.

Respuesta: `chunks` (cada uno con `chunk_id`, `text`, `metadata`, `token_count` y `embedding`) y `stats`:

| Campo | Qué es |
|---|---|
| `total_budgets` / `total_chunks` | presupuestos recibidos y chunks producidos |
| `total_tokens` | tokens de entrada **facturados** por la API (suma de `usage.prompt_tokens`) |
| `counted_tokens` | tokens contados en local con tiktoken antes de llamar |
| `estimated_cost_usd` | `total_tokens` × 0,02 USD por millón (precio de `text-embedding-3-small`) |
| `embedding_model`, `dimensions`, `batches`, `latency_ms` | modelo, dimensión del vector, llamadas a la API y tiempo total |

Códigos: **200** correcto, **422** body inválido (lo gestiona FastAPI con Pydantic), **500** si falta
`OPENAI_API_KEY` o la API de embeddings falla (mensaje genérico al cliente, detalle en los logs).

> El orden importa: la comprobación de la API key vive **dentro del handler**, no en la dependencia. FastAPI
> resuelve las dependencias antes de validar el body, así que exigir la clave desde una dependencia convertía
> en **500** cualquier petición mal formada en un entorno sin clave. Lo detectó el CI, que corre sin secrets.
> Con la clave ausente, un body inválido sigue devolviendo 422 y solo un body válido llega al 500.

### `scripts/compare.py`, reemplazado

El script que medía la similitud coseno entre dos textos sueltos en memoria desapareció en la sesión 08:
lo sustituye [`query_examples.py`](query_examples.py), que mide lo mismo donde importa, contra el corpus ya
persistido y por el camino que usaría una aplicación real. La función `cosine_similarity` sigue en
`app/embedding_pipeline/similarity.py` (la usan el sanity check y los tests); el cálculo en producción lo hace
ahora Postgres con el operador `<=>`.

### El sanity check

[`app/embedding_pipeline/SANITY_CHECK.md`](app/embedding_pipeline/SANITY_CHECK.md) tiene las tres parejas del
paso 7 con sus valores reales y la lectura. Resumen: la estructura es la esperada (cercanos 0,5957 ≫ no
relacionados 0,1920), pero la pareja "cercana" **no llega al 0,6** que sugiere el enunciado y dos textos
genéricos ("Backend services" / "API development") dan 0,5407, casi lo mismo. Los umbrales hay que calibrarlos
sobre el corpus propio.

## Persistencia en PostgreSQL + pgvector (pre-work sesión 08)

Los vectores dejan de vivir en memoria. Cada presupuesto ingestado se guarda como un **documento** con sus
**chunks** (uno por componente, cada uno con su embedding) en una sola transacción, y `POST /search` resuelve
una consulta en lenguaje natural por **distancia coseno en SQL**.

```
POST /embeddings/ingest                               POST /search
  un presupuesto JSON                                   query + k
        ↓                                                   ↓
  documents (1 fila)                              OpenAIEmbedder.embed_one
        ↓  flush → document_id                              ↓
  JSONStructuralChunker → N chunks                   vector de la consulta
        ↓                                                   ↓
  OpenAIEmbedder (un lote)                  SELECT ... ORDER BY embedding <=> :q LIMIT k
        ↓                                                   ↓
  chunks (N filas con embedding) → COMMIT        k chunks + su distancia
```

### Puesta en marcha

```bash
docker compose up postgres -d                                   # Postgres 16 + pgvector
docker compose exec postgres psql -U estimator -d estimator -c "SELECT version();"

docker compose up --build -d                                    # api (8000) + chat (8501)
docker compose run --rm api alembic upgrade head                # crea el esquema
docker compose run --rm api python scripts/ingest_corpus.py     # 15 presupuestos → 73 chunks
docker compose run --rm api python query_examples.py            # las cinco consultas del enunciado
```

Fuera del contenedor es lo mismo con `uv run` (el `DATABASE_URL` por defecto apunta a `localhost:5432`, que
es lo que publica el Postgres de compose):

```bash
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
uv run python scripts/ingest_corpus.py --reset      # --reset vacía documents (los chunks caen en cascada)
uv run python query_examples.py --output output_examples.txt
```

> El enunciado llama `ai_service` al servicio de IA; en este proyecto se llama **`api`** desde la sesión 02.
> El comando del entregable es, por tanto, `docker compose run --rm api python query_examples.py`.

### Los dos endpoints

**`POST /embeddings/ingest`** — un documento por petición:

```bash
curl -s -X POST http://localhost:8000/embeddings/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "data/budgets/budget_2024_014.json",
       "document_type": "historical_budget",
       "content": { "budget_id": "BUD-2024-014", "...": "el presupuesto completo" }}'

{"document_id": 1, "chunks_created": 5, "embedding_dimension": 1536, "ingestion_time_ms": 1328}
```

| Código | Cuándo |
|---|---|
| **200** | documento y chunks persistidos |
| **409** | ya hay un documento con ese `source_path`: `{"detail": "Document already ingested", "document_id": 1}` |
| **422** | el body no valida (falta un campo, sector fuera del vocabulario, campo desconocido) |
| **500** | falta `OPENAI_API_KEY` o la API de embeddings falló |
| **503** | PostgreSQL no responde |

Los **tokens y el coste no están en la respuesta** porque el contrato del enunciado tiene cuatro campos y solo
cuatro. Siguen medidos: van al evento structlog `embedding_ingest_completed`, que es donde el proyecto lleva el
gasto desde la sesión 03 (`total_tokens`, `counted_tokens`, `estimated_cost_usd`, `batches`).

**`POST /search`** — los k chunks más cercanos:

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "REST API with OAuth authentication for fintech sector", "k": 5}'
```

```json
{"query": "...", "k": 5, "search_time_ms": 235,
 "results": [{"chunk_id": 18, "document_id": 4, "chunk_type": "budget_component",
              "content": "[Project: Mobile banking API...] Component: OAuth 2.0 authentication backend...",
              "distance": 0.4108,
              "metadata": {"component_name": "OAuth 2.0 authentication backend", "client_sector": "finance", "...": "..."}}]}
```

### El esquema

Una migración, [`alembic/versions/0001_initial_schema.py`](alembic/versions/0001_initial_schema.py): la
extensión `vector`, dos tablas y cuatro índices, ninguno vectorial.

| `documents` | | `chunks` | |
|---|---|---|---|
| `id` | BIGSERIAL PK | `id` | BIGSERIAL PK |
| `source_path` | TEXT, indexado | `document_id` | BIGINT → `documents.id` **ON DELETE CASCADE**, indexado |
| `document_type` | VARCHAR(50) | `chunk_type` | VARCHAR(50), indexado |
| `ingested_at` | TIMESTAMPTZ DEFAULT now() | `content` | TEXT (el texto que se embebió) |
| `metadata` | JSONB DEFAULT '{}' | `embedding` | **VECTOR(1536)**, nullable |
| | | `metadata` | JSONB DEFAULT '{}', **índice GIN** |
| | | `created_at` | TIMESTAMPTZ DEFAULT now() |

En SQLAlchemy el atributo no puede llamarse `metadata` (está reservado en las clases declarativas), así que los
modelos usan `doc_metadata` / `chunk_metadata` **mapeados explícitamente** a la columna `metadata`, que es el
nombre que fija el enunciado. `alembic/env.py` toma la URL de `DATABASE_URL` y registra el tipo `vector` en el
dialecto (`ischema_names`); sin eso, `alembic check` no reconoce las columnas `vector` y genera migraciones
inconsistentes. Con eso, `alembic check` sale limpio.

### Las cuatro decisiones de schema

**(a) Dos tablas y no una.** Un presupuesto produce N chunks (aquí, de 4 a 6). Con una sola tabla, los 10
campos del presupuesto (cliente, sector, país, año, tecnología, horas totales…) se repetirían en cada fila: 73
copias de 15 presupuestos, y nada que impida que dos chunks del mismo presupuesto digan que el sector es
distinto. Normalizado, cada hecho se guarda una vez y en un solo sitio. Lo que lo decide no es el ahorro de
bytes, es **de quién es cada dato**: el sector es del presupuesto, las horas estimadas son del componente.
Además, `ON DELETE CASCADE` convierte «borrar un presupuesto y sus chunks» en un `DELETE FROM documents`, sin
código que orqueste el borrado ni el riesgo de dejar chunks huérfanos apuntando a un documento que ya no está
(`scripts/verificar.sh` lo comprueba borrando un documento y contando sus chunks antes y después).

**(b) `metadata` como JSONB y no como columnas.** La regla que se aplicó: **tipado lo que la aplicación
consulta siempre y conoce de antemano; JSONB lo que el chunker produce y puede crecer.** `document_type`,
`chunk_type`, `source_path` y las fechas son columnas tipadas porque los endpoints y los índices dependen de
ellas y su ausencia es un error. `client_sector`, `tech_stack`, `complexity`, `estimated_hours`,
`component_name` o `token_count` van en JSONB porque los produce el chunker y la lista no está cerrada: en la
sesión 07 eran nueve campos, el día que el chunker enriquezca los chunks (tags, scope, tecnologías detectadas)
serán más, y con columnas tipadas cada campo nuevo sería una migración. El índice **GIN** es lo que hace que
esto no sea una excusa para no modelar: permite `WHERE metadata @> '{"client_sector": "finance"}'` o
`metadata->>'year' = '2024'` con índice, sin tocar el esquema. El precio que se paga, y que conviene decir en
voz alta: JSONB no valida nada (un `year` puede llegar como texto) y no tiene las restricciones de integridad
de una columna. Por eso lo que **sí** se consulta siempre está tipado.

**(c) `cosine_distance` (`<=>`) y no L2 ni producto interno.** Para `text-embedding-3-small` las tres ordenan
igual: OpenAI devuelve vectores normalizados (la norma medida en el sanity check de la sesión 07 fue 1,0000),
y con norma 1 se cumple que `coseno = producto escalar` y que `euclídea² = 2 − 2·coseno`, que es monótona
decreciente en el coseno. Es decir: **la elección no cambia el ranking**, cambia qué índice podrá usar Postgres
después. Un índice HNSW se construye con una *operator class* concreta (`vector_cosine_ops`,
`vector_l2_ops`, `vector_ip_ops`) y solo se usa si la consulta emplea el operador que le corresponde; si no
coinciden, Postgres **ignora el índice sin avisar** y vuelve al sequential scan con la misma respuesta y mucho
más tiempo. Fijar coseno ahora deja consulta e índice alineados para el directo. Segundo motivo, menor pero
real: la distancia coseno vive en [0, 2] con 0 = idéntico, que es la escala de la literatura RAG y la que
hace legible `output_examples.txt`.

**(d) Sin índice vectorial, a propósito.** Es el **baseline**. Sin índice, Postgres hace sequential scan:
calcula las 73 distancias y ordena, y eso es exacto por construcción (recall 100 %). Un HNSW es una búsqueda
*aproximada*: va mucho más rápido y puede perderse vecinos, y la única forma de saber cuánto gana y cuánto
pierde es tener medido el antes. Con 73 chunks el escaneo completo no es un problema (los
`search_time_ms` de `output_examples.txt` van de 214 a 354 ms, y casi todo eso es la llamada de embedding de
la consulta, no el SQL), así que el índice ahora mismo no compraría nada medible —construirlo sería optimizar sin
tener el problema. `scripts/verificar.sh` y `tests/test_vector_store.py` **fallan si alguien añade un índice
vectorial**, para que la ausencia siga siendo una decisión y no se convierta en un descuido.

### Las cinco consultas

[`query_examples.py`](query_examples.py) lanza las cinco del enunciado contra `/search` y escribe
[`output_examples.txt`](output_examples.txt). Cada consulta declara **qué habría que observar** si el pipeline
está bien, y el informe cierra con una tabla comparativa. Resultado real sobre los 73 chunks:

| Consulta | Mejor | Peor | Margen | Primer resultado |
|---|---|---|---|---|
| Componente directo conocido | 0,4109 | 0,4879 | 0,0770 | OAuth 2.0 authentication backend |
| Reformulación semántica | 0,4074 | 0,4740 | 0,0665 | OAuth 2.0 authentication backend |
| Dominio distinto | 0,5317 | 0,5751 | 0,0434 | Online appointment booking |
| Consulta ambigua | 0,5810 | 0,6054 | 0,0244 | ERP integration (SAP Business One) |
| Consulta muy específica | 0,6079 | 0,6904 | 0,0824 | Telemetry ingestion |

Reproducibilidad, medida: repitiendo el flujo entero desde un volumen de Postgres vacío (`docker compose
down -v`, migración, ingesta, consultas), los **`chunk_id` salen idénticos** —de ahí el `TRUNCATE ... RESTART
IDENTITY` del `--reset`— y el **ranking es el mismo en las cinco consultas**, pero las distancias se mueven en
el **cuarto decimal** (0,4108 frente a 0,4109; 0,6052 frente a 0,6054). La API de embeddings no es
bit-exacta entre llamadas, así que los vectores reingestados no son idénticos a los anteriores. Es una cifra
pequeña y aquí no cambia nada, pero conviene tenerla presente antes de comparar distancias de dos ingestas
distintas o de fijar un umbral con cuatro decimales de precisión.

Cuatro lecturas, que es lo que se lleva al directo:

- **Los embeddings capturan significado, no palabras.** La reformulación («secure backend service with
  token-based access control for banking applications») no comparte *ni una* palabra de contenido con la
  consulta directa («REST API development with JWT authentication…»), y recupera el **mismo chunk en primer
  lugar** con una distancia incluso menor (0,4074 frente a 0,4109). Es la prueba que justifica todo el
  pipeline: con búsqueda por palabras clave, esta consulta no habría encontrado nada.
- **La distancia es la única señal de «aquí no hay nada».** El top-k siempre devuelve k filas: la consulta de
  otro dominio (reservas de restaurante) devuelve cinco resultados con cara de buenos, y lo único que delata
  que ninguno sirve es que la mejor distancia salte de 0,41 a 0,53. Un RAG sobre esto necesita un **umbral**,
  no un `LIMIT`.
- **La consulta ambigua no tiene ganador y se nota en el margen.** «integration with external system» produce
  cinco distancias dentro de 0,0244 (0,5810-0,6054): el ranking es casi arbitrario. El margen entre el primero
  y el quinto es un indicador de confianza mucho más honesto que la posición.
- **La consulta específica expone el límite del recuperador.** No hay nada de monolito a microservicios con
  Kubernetes en el corpus, y el sistema responde «Telemetry ingestion» a 0,6079: recupera *lo más cercano*,
  que no es *lo relevante*. La respuesta correcta («no tengo nada de eso») requiere un umbral y, en el
  directo, filtros y búsqueda híbrida.

### Lo que no se hizo, por disciplina de scoping

El enunciado reserva cuatro temas para la sesión en vivo y ninguno está aquí: **índices vectoriales** (HNSW,
IVFFlat), **filtros por metadata** en el `WHERE` (la metadata ya está en JSONB con GIN, pero el endpoint no
filtra), **búsqueda híbrida** (full-text + vector) y **tuning** de `shared_buffers`, `maintenance_work_mem` o
`ef_search` (todo en valores por defecto). El test `test_search_statement_uses_the_cosine_operator_ordered_and_limited`
comprueba incluso que la consulta **no lleva `WHERE`**.

Lo que sí se añadió sobre el mínimo, y por qué: el **503** cuando Postgres no responde (sin él, una base caída
devolvía un 500 genérico indistinguible de un fallo del embedder), el **rollback explícito** ante un fallo del
embedder (es lo que hace cierta la atomicidad que pide el enunciado, y hay un test que lo comprueba), y
`scripts/ingest_corpus.py` (el contrato es un documento por petición, así que ingestar el corpus del curso son
15 llamadas; hacerlas a mano con `curl` no era razonable).

## Huecos del curso, cerrados

Cuatro cosas que el material enseña y que el proyecto no tenía. Todas son **aditivas**: el comportamiento
por defecto no cambia, para no invalidar las entregas ni el baseline del stress.

### 1. Salida estructurada (sesión 04)

*Un producto no consume texto, consume datos.* El endpoint devolvía Markdown en un campo de texto y quien
lo quisiera usar tenía que parsearlo.

```bash
curl -X POST localhost:8000/api/v1/estimate -H "Content-Type: application/json" -d '{
  "transcription": "...", "hourly_rate_eur": 55, "response_format": "structured"}'
```

`response_format` acepta `markdown` (por defecto, comportamiento histórico), `structured` (objeto tipado) o
`both`. En los dos últimos el modelo responde contra un **JSON Schema** y el objeto se valida antes de salir
del servicio: `app/schemas/structured.py` define horas, coste, equipo, supuestos, riesgos y preguntas.

Lo que de verdad cambia el resultado no es el esquema, es la **validación de coherencia**: los validadores
comprueban que las líneas sumen el subtotal, que la contingencia cuadre y que el coste sea horas × tarifa.
Cuando algo de eso falla, el error vuelve al modelo como un mensaje más y se le pide que lo corrija, hasta
tres intentos. La respuesta dice en `structured_info` cuántos intentos hicieron falta y qué se corrigió.

> **Por qué no Instructor**, que es lo que recomienda el material: su última versión fija `jiter<0.15` y el
> SDK de OpenAI del proyecto exige `jiter>=0.16`, así que no hay resolución posible. Se implementa a mano lo
> que Instructor hace por dentro, con las vías nativas de cada proveedor: `chat.completions.parse` en OpenAI
> y `output_config.format` en Anthropic.

### 2. Guardrails de entrada y salida (sesión 04)

La forma no garantiza el contenido. `app/services/guardrails.py` cubre la mitad semántica de la matriz del
material (la sintáctica ya la cubrían `EstimationRequest` y `StructuredEstimation`):

| | Entrada | Salida |
|---|---|---|
| Qué mira | moderación de OpenAI, patrones de inyección de prompt, PII evidente | fuera de dominio, cifras implausibles, desglose pobre, confianza baja sin preguntas |
| Política | `BLOCK` la inyección y la moderación; `DEGRADE` la PII | `DEGRADE` todo: el usuario ya pagó la llamada |

Cada regla **declara su política**, que es la regla que el material pide no dejar implícita. Y el modo por
defecto es `log`: evalúa, registra el evento `guardrail_triggered` y **no bloquea nada**. Es el despliegue
que recomienda el curso, una o dos semanas mirando falsos positivos antes de poner `GUARDRAILS_MODE=enforce`.

Los guardrails de entrada corren **antes de la caché**: servir un acierto de caché saltándose la validación
convertiría la caché en un vector de inyección.

### 3. Caché semántica por embeddings (sesión 04)

`SEMANTIC_CACHE_BACKEND=embeddings` cambia la comparación de Jaccard sobre palabras a coseno sobre
embeddings, que es lo que reconoce *significados* y no solo vocabulario compartido.

El umbral **no se puede copiar del material**. Medido sobre peticiones reales de este dominio
([`docs/calibracion-cache-semantica.md`](docs/calibracion-cache-semantica.md)), el 0,90-0,93 que recomienda
el curso solo reconoce 1 de cada 5 reformulaciones:

| Umbral | Reformulaciones que acierta | Peticiones distintas que serviría mal |
|---|---|---|
| 0,90 (el del material) | 1 de 5 | 0 de 5 |
| **0,75 (el que se ha fijado)** | **4 de 5** | **0 de 5** |
| 0,55 | 5 de 5 | 1 de 5 |

Y las dos clases **se solapan**: la peor reformulación equivalente da 0,594 y la peor pareja distinta 0,627,
así que ningún umbral las separa del todo. Mismo hallazgo que el sanity check de la sesión 07.

### 4. LiteLLM como wrapper de proveedores (sesión 03)

`LLM_BACKEND=litellm` enruta las llamadas por el agregador en vez de por los SDK directos. *Lo que un ORM
hace con las bases de datos, LiteLLM lo hace con los LLMs*: el system prompt pasa a ser un mensaje más y la
respuesta se lee igual venga de donde venga.

Se deja como **backend opcional** y no como sustitución: los SDK directos son los que usaron las entregas de
las sesiones 02 a 07, y cambiarlos invalidaría la comparación con el baseline del stress. Comprobado que los
dos caminos dan el mismo resultado sobre el mismo prompt (3.314 tokens de entrada y la misma estimación).

> El material avisa de que las versiones 1.82.7 y 1.82.8 de LiteLLM en PyPI estuvieron comprometidas en
> marzo de 2026. La dependencia se fija por encima de esas dos y `uv.lock` deja la versión exacta.

### Lo que no se ha tocado

El streaming del formulario sigue en los SDK; el modo `markdown`, la ventana deslizante, las anclas, el
resumen y el tier funcionan exactamente igual que en las entregas. 222 tests en verde.

## Interfaz: conversación por turnos (Streamlit)

`streamlit_app.py`, en la raíz del proyecto, es la interfaz del estimador. En la sesión 03 era un chat; en
la 04 el chat dejó de ser la interfaz principal y se sustituyó por un **formulario** cuyos campos son
exactamente los de `EstimationRequest`, el mismo modelo que valida el endpoint. En el pre-work de la 05 el
formulario se mantiene, pero cada envío es un **turno** dentro de una **sesión** de la API:

- Al cargar la página crea la sesión (`POST /api/v1/sessions`) y guarda el `session_id` en
  `st.session_state`. El botón **"Nueva conversación"** vuelve a llamarlo y resetea el estado.
- Campo de texto para la transcripción (o lo nuevo del turno), **selector múltiple de ficheros** (PDF,
  Word, texto) y los parámetros tipados en un desplegable. Un selector carga las transcripciones de ejemplo;
  las de "Panadería" forman la conversación de tres turnos de arriba, y los adjuntos de ejemplo están en
  `transcripciones/adjuntos/`.
- Los turnos se van apilando en la página con su estimación, la transcripción enviada, los adjuntos leídos
  y las métricas (mensajes al LLM, tokens, coste, tiempo).
- La **barra lateral** muestra la **memoria del proyecto** (`project_metadata`, con su JSON) y el
  **historial efectivo** (los mensajes que viajarían al LLM, tamaño de cada uno, turnos en ventana y
  descartados): es la forma de ver la separación entre memoria e historial.
- Habla con la API por HTTP (`app/clients/estimator_api.py`, `API_BASE_URL`). Sin API arrancada muestra
  el error y se detiene; si la API se reinició y perdió la sesión, crea otra.

```bash
uv run uvicorn app.main:app --reload       # la API, en otra terminal
uv run streamlit run streamlit_app.py      # abre http://localhost:8501
```

Capturas de una conversación real de tres turnos (el tercero con el PDF y el Word adjuntos) en
`docs/capturas/`: [`sesion-05-memoria-del-proyecto.jpg`](docs/capturas/sesion-05-memoria-del-proyecto.jpg)
(panel de memoria tras el turno 3: tecnologías de los adjuntos, equipo, última estimación, documentos),
[`sesion-05-historial-y-metricas.jpg`](docs/capturas/sesion-05-historial-y-metricas.jpg) (ventana de 6 turnos,
3 en ventana, 6 mensajes al LLM, 6.476 tokens de entrada) y [`sesion-05-turno-1.jpg`](docs/capturas/sesion-05-turno-1.jpg).

Por qué un formulario y no un chat: pedir una estimación no es una conversación, es una operación con
entradas conocidas. Con un chat el usuario tiene que adivinar qué decirle al modelo (¿la tarifa?, ¿el
plazo?, ¿cuánto detalle?) y cada petición es distinta e irrepetible. Con un formulario las entradas son
explícitas, validadas antes de gastar un token, y dos personas con los mismos datos obtienen el mismo
prompt. El chat se puede añadir después como capa secundaria ("pregúntale a la estimación"), pero no
como interfaz principal.

Qué hace:

- **Formulario** (`st.form`): transcripción, nombre y tipo de proyecto, tarifa, plazo objetivo, enfoque y
  nivel de detalle. Los valores por defecto salen de `EstimationRequest`, no están duplicados en la UI.
  Un selector carga las transcripciones de ejemplo del proyecto.
- **Validación tipada antes de llamar al LLM**: el envío construye un `EstimationRequest`; si Pydantic lo
  rechaza (transcripción corta, tarifa fuera de rango...) se muestran los errores y no se hace la llamada.
- **Streaming**: la estimación se va escribiendo con `st.write_stream` (`EstimationStream`, en
  `app/services/llm_service.py`), igual que en la sesión 03. Cada envío es una llamada independiente.
- **Prompt enviado**: bajo cada estimación, un desplegable muestra el system prompt y el mensaje de
  usuario tal como se enviaron, con la versión de plantilla usada. Se puede descargar la estimación en `.md`.
- **Barra lateral**: proveedor (solo los que tienen API key), modelo y parámetros; selector de **versión de
  plantilla** con su descripción, changelog y el texto crudo de `system.md.j2` y `user.md.j2`; el contexto
  CAG inyectado; y las métricas de la última llamada: tokens de entrada y salida, tiempo, coste y cuántos
  caracteres son contexto fijo y cuántos variables.
- **API key**: se lee de `.env` (igual que la API) o, si no está ahí, de `.streamlit/secrets.toml` con
  `st.secrets`. Sin API key el botón de envío aparece deshabilitado con un aviso.

Tests sin llamadas reales: `tests/test_streaming.py` (el servicio de streaming, que sigue usándose en modo
transaccional) y `tests/test_streamlit_app.py` (la app completa con `streamlit.testing.v1.AppTest` contra la
API real en el mismo proceso, con el cliente HTTP sustituido por el `TestClient` de FastAPI y el LLM por un
doble: crea la sesión al cargar, encadena tres turnos y comprueba que el historial y la memoria llegan al
LLM y al panel, "Nueva conversación", la validación sin llamada y el error sin API).

## Prompts como plantillas versionadas

El texto de los prompts ya no está en `llm_service.py`: vive en `prompts/estimacion/<versión>/` como
artefactos de software con su propia versión, metadatos y changelog (`prompts/README.md` explica las
reglas). Cada versión son tres ficheros:

| Fichero | Qué es |
|---|---|
| `template.toml` | Nombre, versión, descripción, fecha, changelog y **contrato de variables** de cada plantilla |
| `system.md.j2` | System prompt (Jinja2): rol, reglas, formato de salida y los ejemplos CAG |
| `user.md.j2` | Mensaje de usuario (Jinja2): parámetros de la petición y transcripción |

`app/services/prompts.py` es el registro: lista versiones, carga una (`load_template`), y la renderiza con
Jinja2 en modo estricto. Fallos que se detectan antes de llamar al LLM: versión inexistente (`422` en la
API), variable declarada en `[variables]` que el código no pasa, y variable usada en la plantilla sin
declarar.

Versiones:

- **`v1`** es el prompt original de las sesiones 02 y 03, migrado sin cambiar una palabra. Sirve para
  comparar y para reproducir estimaciones antiguas. Mezclaba un dato de la petición (la tarifa) en el
  system prompt.
- **`v2`** (por defecto) separa el **contexto fijo** (rol, reglas, dos formatos de salida, ejemplos) en el
  system prompt, idéntico en todas las llamadas, de los **datos de la petición** (nombre, tipo, tarifa,
  plazo, enfoque, nivel de detalle, transcripción) en el mensaje de usuario. Un prefijo estable es lo que
  permite cachear el prompt en el proveedor, y es lo que hace que `system_prompt_chars` mida de verdad el
  contexto CAG.

Misma transcripción (`transcripciones/reunion-landing-page-parametros.json`), `gpt-4o-mini`, tres peticiones
reales el 2026-09-28:

| Petición | Plantilla | System (caracteres) | User (caracteres) | Tokens entrada | Tokens salida | Coste (USD) |
|---|---|---:|---:|---:|---:|---:|
| Parámetros por defecto, nivel `completa` | `v2` | 10.414 | 1.260 | 2.931 | 425 | 0.000695 |
| Enfoque `equilibrada`, nivel `resumen` | `v2` | 10.414 | 1.267 | 2.936 | 310 | 0.000626 |
| Solo la transcripción | `v1` | 7.456 | 421 | 1.954 | 325 | 0.000488 |

Qué se ve:

- `v2` cuesta unos 1.000 tokens de entrada más que `v1`: el system prompt describe los dos formatos y las
  reglas de cada parámetro, y el mensaje de usuario lleva los parámetros y unos recordatorios. Es el
  precio de una salida consistente: con `v2` la estimación trae subtotal, contingencia según el enfoque,
  total cuadrado y la valoración del plazo objetivo; con `v1` cada respuesta decide sola qué incluir.
- El nivel `resumen` recorta la salida (310 frente a 425 tokens) sin tocar la entrada.
- El system prompt de `v2` es idéntico en las tres llamadas: es la parte cacheable.
- Al iterar el prompt salieron dos cosas que solo se ven probando: `gpt-4o-mini` ignoraba la contingencia y
  el plazo cuando solo estaban en el system prompt (se resolvió con los recordatorios por petición en el
  mensaje de usuario) y calculaba mal los porcentajes (se resolvió pidiendo un subtotal explícito antes de
  la contingencia). Ambos cambios están en el changelog de `v2`.

Cómo se cambia un prompt: nunca editando una versión publicada. Se crea `v3/`, se documenta el cambio en su
changelog, se prueba pidiéndola con `prompt_version: "v3"` y, cuando convence, se activa con
`PROMPT_VERSION=v3` en `.env`. `GET /api/v1/prompts` y el selector de la barra lateral la listan
automáticamente: no hay que tocar código.

## Tokens y coste

Cada respuesta incluye los tokens de entrada y de salida que reporta el proveedor y el coste de la llamada
en USD, calculado con el precio de lista del modelo (`app/services/pricing.py`):

| Modelo | Entrada (USD / millón de tokens) | Salida (USD / millón de tokens) |
|---|---|---|
| `gpt-4o-mini` | 0.15 | 0.60 |
| `gpt-4o` | 2.50 | 10.00 |
| `claude-haiku-4-5` | 1.00 | 5.00 |
| `claude-sonnet-5` | 2.00 | 10.00 |
| `claude-opus-5` | 5.00 | 25.00 |

Los precios cambian: revísalos en las webs de [OpenAI](https://openai.com/api/pricing/) y
[Anthropic](https://www.anthropic.com/pricing). Para un modelo que no esté en la tabla, o para forzar otro
precio, define `LLM_INPUT_PRICE_PER_MTOK` y `LLM_OUTPUT_PRICE_PER_MTOK` en `.env`. Si no hay precio
conocido, `usage.cost` viene a `null`.

En arquitectura CAG el system prompt con los ejemplos viaja **en todas las llamadas**: con la plantilla `v2`
son unos 10.400 caracteres, alrededor de 2.600 tokens (con `v1`, 7.500 y 1.900), y es la mayor parte de la
entrada. Los parámetros y la transcripción solo añaden lo que ocupen. Por eso, si el objetivo es abaratar,
lo que pesa es el contexto fijo, no lo que escriba el cliente; y por eso conviene que ese contexto sea
idéntico entre llamadas, para poder cachearlo.

## Comparar una transcripción pobre con una detallada

La calidad de la estimación depende directamente de la calidad de la transcripción, y el coste en tokens
apenas cambia. Para verlo hay dos transcripciones del **mismo proyecto**, una tienda online para una
panadería artesanal:

| Archivo | Qué contiene |
|---|---|
| `transcripciones/panaderia-descripcion-pobre.json` | Tres frases vagas: quiere vender por internet, que esté lista pronto y que no sea cara. |
| `transcripciones/panaderia-descripcion-detallada.json` | Reunión completa: catálogo, stock diario, franjas de recogida, reparto, pagos, pedidos recurrentes, panel del obrador, idiomas, plazo y presupuesto. |

### Con el script de comparación

```bash
cd estimador-cag
docker compose up --build -d          # o: uv run uvicorn app.main:app --reload
uv run scripts/comparar.py            # pobre vs. detallada por defecto
uv run scripts/comparar.py transcripciones/a.json transcripciones/b.json   # cualquier pareja o lista
docker compose down
```

Salida real del 2026-09-16 con `gpt-4o-mini` y temperatura 0.2:

```text
Comparativa de transcripciones · gpt-4o-mini (openai) · precios: 0.15 / 0.6 USD por millón de tokens (entrada / salida)

Caso                               Caract.  Tok. entrada  Tok. salida  Tok. total   Coste USD   Estimación
----------------------------------------------------------------------------------------------------------
panaderia-descripcion-pobre            173         1.929          341       2.270    0.000494   225 h · 7 tareas · 2 preguntas
panaderia-descripcion-detallada      2.672         2.581          433       3.014    0.000647   380 h · 10 tareas · 3 preguntas
Diferencia (2ª - 1ª)                +2.499          +652          +92        +744   +0.000153   +155 h

Contexto fijo (system prompt con los ejemplos CAG): 7.458 caracteres ≈ 1.884 tokens en TODAS las llamadas (98% de la entrada del 1º caso, 73% del 2º).
Ahorro en tokens de «panaderia-descripcion-pobre» frente a «panaderia-descripcion-detallada»: 0.000153 USD por llamada (23.6%), es decir, 0.15 USD por cada 1.000 estimaciones.
Diferencia entre las estimaciones obtenidas: 155 horas (7.750 € a 50 €/hora). Ese es el orden de magnitud de lo que está en juego frente al ahorro en tokens.
```

Lectura de los números:

- **Tokens de entrada:** la detallada suma 652 tokens más (la transcripción es 15 veces más larga), pero
  el contexto fijo de los ejemplos ya son unos 1.884 tokens en las dos. La transcripción pobre solo
  aporta un 2 % de su entrada.
- **Tokens de salida:** casi iguales (341 frente a 433): el formato de salida está fijado por el prompt,
  así que la estimación ocupa lo mismo sea buena o mala.
- **Ahorro:** la pobre cuesta 0.000153 USD menos por llamada, un 23.6 %. En 1.000 estimaciones son
  0.15 USD. A cambio, la estimación describe un proyecto inventado y se desvía 155 horas (7.750 €) de la
  detallada. El ahorro en tokens no compensa: la palanca de coste está en el contexto fijo y en el modelo,
  no en recortar el requerimiento.
- Las horas varían un poco entre ejecuciones con temperatura 0.2 (230/410 en una tirada, 225/380 en
  otra); los tokens de entrada son idénticos porque el prompt es determinista.

### A mano, con curl

```bash
# Transcripción pobre
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/panaderia-descripcion-pobre.json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["estimation"]); print(d["usage"])'

# Transcripción detallada
curl -s -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @transcripciones/panaderia-descripcion-detallada.json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["estimation"]); print(d["usage"])'
```

El `| python3 ...` solo imprime la estimación y el bloque de tokens y coste; sin él se ve el JSON completo.
La salida completa de ambas estimaciones y el análisis de lo que el modelo acierta y omite está en
[`docs/comparativa-pobre-vs-detallada.md`](docs/comparativa-pobre-vs-detallada.md).

Qué observar en el contenido:

- Con la transcripción pobre el modelo inventa un alcance estándar y lo estima: la cifra parece
  razonable, pero no describe el proyecto real.
- Con la detallada el desglose refleja lo hablado, pero todavía se deja cosas: no avisa de que supera el
  presupuesto de 12.000 € y el plazo de 8 semanas, y omite el reparto por radio, el bilingüismo, Bizum y
  la exportación a Excel. Es el punto de partida para iterar el prompt en la sesión en vivo.

Para añadir un caso nuevo: escribe la transcripción en `transcripciones/<nombre>.txt`, ejecuta
`scripts/generar-json.sh` para crear el `.json` equivalente, compáralo con
`uv run scripts/comparar.py transcripciones/<a>.json transcripciones/<b>.json`, y anota el resultado en
`docs/comparativa-pobre-vs-detallada.md`.

## Tests y verificación automática

```bash
uv run pytest -q          # tests con el LLM sustituido por dobles (no gasta créditos, no necesita Postgres)
scripts/verificar.sh      # estructura + tests + arranque real + llamada real si hay API key
```

Los 251 tests corren **sin base de datos y sin API key**: la sesión de SQLAlchemy se sustituye por un doble
(`tests/test_vector_store.py`) que registra el orden de las operaciones, y el SQL de la búsqueda se comprueba
compilándolo al dialecto de Postgres en vez de ejecutándolo. Lo que sí necesita un Postgres de verdad
—`alembic upgrade head`, `alembic check`, el 409, la cascada y una búsqueda real— lo verifica
`scripts/verificar.sh` en su paso 4, y lo omite con un aviso si la base no responde.

Los tests de la sesión 05 están en `tests/test_sessions_api.py` (integración con `httpx.AsyncClient` sobre la
app ASGI), `tests/test_sessions.py` (ventana, memoria y store), `tests/test_attachments.py` (extracción) y
`tests/test_metadata_extractor.py`. Los tres que pide el enunciado:

| Test | Qué comprueba |
|---|---|
| `test_dos_turnos_en_la_misma_sesion_actualizan_la_memoria` | Dos peticiones enlazadas: la memoria acumula tecnologías y actualiza alcance y horas, el system prompt del turno 2 lleva los hechos del turno 1 (y su prefijo es idéntico), y el historial del turno 1 viaja como turnos anteriores |
| `test_un_pdf_adjunto_cambia_la_estimacion` | Misma transcripción con y sin un PDF que menciona Stripe: el texto llega al prompt entre separadores, la estimación incorpora la tarea de pagos y `last_total_hours` cambia (84 → 108) |
| `test_ocho_turnos_respetan_la_ventana_deslizante` | Ocho turnos con `SESSION_MAX_TURNS=3`: en ninguna llamada viajan más de 3 mensajes de usuario ni más de 2 de asistente, el system prompt va siempre y solo una vez, los turnos 1-5 desaparecen y la memoria conserva el nombre fijado en el turno 1 |

El doble del LLM (`tests/fakes.py`) genera la estimación a partir del prompt que recibe (toma el nombre del
proyecto de los parámetros o del bloque `<project_metadata>`, y añade una tarea de pagos si ve "Stripe"),
así que los tests son deterministas y comprueban el mecanismo, no la calidad del modelo.

`scripts/verificar.sh` comprueba, en orden:

1. Que existen todos los ficheros de la estructura del ejercicio (incluidas las plantillas `v1` y `v2`),
   que `.env` está en `.gitignore` y no versionado, que no hay API keys en el código, que el texto del
   prompt no está en el código, que `streamlit_app.py` no usa chat, que **ninguna migración crea un índice
   vectorial** (el baseline de la sesión 08) y que `alembic.ini` no lleva credenciales escritas.
2. Que pasan los tests.
3. Que el servicio arranca y responden `GET /health` (200), `GET /docs` (200), que `POST /api/v1/estimate`
   está en el OpenAPI y valida el body tipado (422 con transcripción vacía, con un enumerado inválido y con
   una versión de plantilla inexistente), que `GET /api/v1/prompts` lista `v1`, `v2` y `v3`, que
   `POST /api/v1/sessions` crea una sesión y el endpoint multipart devuelve 404/422/415 cuando toca, que
   `POST /embeddings/ingest` y `POST /search` están publicados y validan su body, que la migración renderiza
   el esquema esperado en modo offline (`alembic upgrade head --sql`, sin necesidad de base de datos) y que
   la interfaz de Streamlit arranca.
4. Si hay API key configurada, que `POST /api/v1/estimate` con parámetros devuelve una estimación real,
   que una sesión de dos turnos (el segundo con el PDF de ejemplo) funciona y actualiza la memoria, y —si
   PostgreSQL responde— que la migración se aplica, que `alembic check` no detecta divergencias, que una
   ingesta persiste el documento y la segunda devuelve 409, que borrar el documento borra sus chunks en
   cascada y que `POST /search` devuelve resultados ordenados por distancia.

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

Checklist de la sesión 04 (pre-work):

- [x] El chat ya no es la interfaz principal
- [x] Un formulario con parámetros tipados lo sustituye (`EstimationRequest`, compartido con la API)
- [x] El prompt está fuera del código, en templates versionados (`prompts/estimacion/v1`, `v2`)
- [x] El endpoint acepta la nueva estructura (`POST /api/v1/estimate` con parámetros; `GET /api/v1/prompts`)

Checklist del pre-work de la sesión 05:

- [x] `POST /api/v1/sessions` crea una sesión y devuelve un `session_id` (UUID v4)
- [x] `POST /api/v1/sessions/{session_id}/estimate` acepta multipart/form-data con transcripción y adjuntos opcionales y devuelve la misma respuesta que `/estimate` (más el estado de la sesión)
- [x] Tras varios turnos el LLM mantiene el proyecto en curso (conversación real de tres turnos, mismo nombre, estimación actualizada)
- [x] `project_metadata` se actualiza entre turnos y se ve en el panel lateral y en `GET /api/v1/sessions/{id}`
- [x] El historial respeta la ventana deslizante (`SESSION_MAX_TURNS`, test de 8 turnos)
- [x] README: camino B de adjuntos y por qué; extracción heurística de `project_metadata` y por qué
- [x] Tests del paso 7 en verde (`uv run pytest -q`)

Checklist del pre-work de la sesión 07:

- [x] Módulo `app/embedding_pipeline/` con `__init__.py`, `schemas.py`, `chunker.py`, `embedder.py` y `router.py`
- [x] `JSONStructuralChunker.chunk(budgets)`: un componente = un chunk, con cabecera del presupuesto padre, `chunk_id` = `{budget_id}::{component_id}`, metadata filtrable y `token_count` con tiktoken
- [x] `OpenAIEmbedder` con `embed_one` y `embed_many`: `text-embedding-3-small` a 1536 dimensiones, lotes de 100, reintentos 1 s/2 s/4 s ante rate limit, logging structlog por lote y coste estimado
- [x] `POST /embeddings/ingest` registrado bajo `/embeddings`, visible e invocable desde `/docs` (200 / 422 / 500)
- [x] Similitud coseno calculada a mano y verificada contra el sanity check (`app/embedding_pipeline/similarity.py`)
- [x] `app/embedding_pipeline/SANITY_CHECK.md` con los tres valores y el comentario
- [x] README con las instrucciones de ejecución; `pyproject.toml` con `tiktoken` añadido
- [x] Extra (no exigido): 11 tests del pipeline sin gastar créditos (`tests/test_embedding_pipeline.py`)

### Pre-work sesión 08 (PostgreSQL + pgvector)

- [x] `docker-compose.yml` con el servicio `postgres` (`pgvector/pgvector:pg16`, healthcheck, volumen) y `api` esperando a que esté sano
- [x] `SELECT version();` responde dentro del contenedor antes de seguir (paso 1 del enunciado)
- [x] Alembic configurado con plantilla async, URL desde `DATABASE_URL` y el tipo `vector` registrado en el dialecto
- [x] Migración `0001_initial_schema` con la extensión, `documents`, `chunks` y los cuatro índices no vectoriales
- [x] `alembic upgrade head` aplicado y `alembic check` limpio (modelos y esquema coinciden)
- [x] `POST /embeddings/ingest` persiste documento + chunks en una sola transacción y devuelve los cuatro campos del contrato
- [x] Un `source_path` repetido devuelve **409** con `detail` y `document_id` al mismo nivel
- [x] `POST /search` devuelve los k chunks más cercanos por `cosine_distance` (`<=>`), resuelto en SQL
- [x] `query_examples.py` ejecutable con `docker compose run --rm api python query_examples.py`
- [x] `output_examples.txt` con el output real contra el corpus (15 documentos, 73 chunks)
- [x] Sección del README justificando las cuatro decisiones (dos tablas, JSONB, coseno, sin índice)
- [x] Sin índices vectoriales, sin filtros por metadata, sin híbrida y sin tuning (lo comprueban el test y `verificar.sh`)
- [x] Extra (no exigido): 34 tests de persistencia y búsqueda sin Postgres ni créditos (`tests/test_vector_store.py`)

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
- **Campos extra en la respuesta**: `usage` (tokens y coste), `prompt` (tamaño del contexto fijo y de la
  transcripción) y `generated_at`. El coste se calcula en el servidor con una tabla de precios de lista
  para que cada llamada muestre lo que ha costado sin depender de la consola del proveedor.
- **Docker opcional**: el ejercicio no lo pide, pero el stack del curso corre en contenedores. El
  `Dockerfile` y el `docker-compose.yml` no cambian nada del código: la misma app corre con `uv run` o
  en contenedor.
- **Streaming síncrono para Streamlit**: Streamlit ejecuta el script de forma síncrona, así que
  `EstimationStream` usa los clientes síncronos de los SDK y expone un iterable que `st.write_stream`
  consume directamente. El endpoint REST sigue usando la versión asíncrona; ambos comparten la plantilla
  renderizada, la traducción de errores y el cálculo de coste.
- **Un solo contrato de entrada**: `EstimationRequest` (Pydantic, `extra="forbid"`) lo usan el endpoint y
  el formulario. La validación es una y está antes de la llamada al LLM: un parámetro inválido cuesta cero
  tokens. Los valores cerrados son `Literal`, así que Swagger los documenta y el formulario los ofrece.
- **Prompts como artefactos**: carpeta por versión, metadatos en TOML (`tomllib` viene con Python 3.11),
  plantillas Jinja2 con `StrictUndefined`. Una versión publicada no se edita; se crea la siguiente. La
  petición puede pedir una versión concreta para comparar, y la respuesta dice cuál se usó.
- **System prompt estable, datos en el mensaje de usuario** (`v2`): el contexto CAG es idéntico en todas
  las llamadas, cacheable, y la parte variable es pequeña. `v1` se conserva para ver la diferencia.
- **Dos modos**: `POST /api/v1/estimate` sigue siendo transaccional (una petición, una estimación, sin
  estado) y `POST /api/v1/sessions/{id}/estimate` es conversacional. Comparten `EstimationRequest`, las
  plantillas, la llamada al proveedor y el cálculo de coste; solo cambia quién construye el prompt.
- **Historial y memoria separados** (`ConversationHistory` / `ProjectMetadata`): el historial es texto que
  se recorta con ventana deslizante; la memoria son hechos tipados que se regeneran en el system prompt. El
  system prompt no se guarda en el historial para que nunca salga de la ventana.
- **La memoria va al final del system prompt** (`v3`) para no romper el prefijo cacheable.
- **Adjuntos por extracción local** (camino B) con `pypdf` y `python-docx`, concatenados a la transcripción
  con separadores; límites de ficheros, bytes y caracteres por configuración.
- **`project_metadata` por heurística** sobre el formato fijo de la estimación; sin llamadas extra. El
  nombre del proyecto se fija una vez y solo lo cambia el formulario.
- **Sesiones en memoria del proceso** (diccionario con tope y expulsión de la menos reciente), un turno a la
  vez por sesión y rollback del turno si el LLM falla.
- **La interfaz habla con la API por HTTP**: el estado vive en la API, que es la única que sabe de sesiones.
  Se pierde el streaming en el modo conversacional (ver "Límites y decisiones").

Checklist del pre-work de la sesión 06:

- [x] Cada `POST /api/v1/sessions/{id}/estimate` emite un evento `turn_observed` con los 13 campos (structlog, JSON) y lo devuelve en la respuesta y en `GET /api/v1/sessions/{id}`
- [x] `uv run python -m evals.stress.run --http http://localhost:8000` corre de punta a punta y deja `evals/stress/results.csv` con ≥ 50 filas (900 en la tirada completa)
- [x] `evals/stress/metrics.py` expone `LatencyBudgetMetric`, `CostBudgetMetric` y `MemoryDriftMetric` con tests en verde (`tests/test_stress_metrics.py`)
- [x] `evals/stress/REPORT.md` con la tabla resumen, las tres curvas en tabla y los dos párrafos de lectura con afirmaciones cuantitativas (generado por `evals/stress/report.py`)
- [x] Todo en el repo: código, escenarios, métricas, runner, reporte y CSV

## Próximos pasos (sesión en vivo)

- Sesión 07 (directo): otras estrategias de chunking sobre este mismo pipeline (recursive, semantic,
  hierarchical, late chunking, Contextual Retrieval con LLM) y comparación con la estructural; comparativa
  entre modelos de embeddings y Matryoshka (dimensión reducida). Preguntas que deja el sanity check: cómo
  calibrar el umbral cuando el suelo del corpus está en 0,43, si conviene top-k en lugar de umbral absoluto, y
  qué hacer con los componentes de descripción anormalmente larga.
- Sesión 08 (directo), sobre el baseline que deja este pre-work: **índice vectorial** HNSW con
  `vector_cosine_ops` y comparación contra el sequential scan medido (214-354 ms con 73 chunks, casi todo
  embedding de la consulta); **filtros por metadata** en el `WHERE` comparando planes de ejecución (la
  metadata ya está en JSONB con índice GIN, lista para usarse); **búsqueda híbrida** full-text + vector; y
  **tuning** (`shared_buffers`, `maintenance_work_mem`, `ef_search`), hoy todo en valores por defecto.
  Preguntas que deja el output: qué umbral de distancia separa «esto sirve» de «aquí no hay nada» cuando el
  salto medido entre dominio propio y ajeno es 0,41 → 0,53, y si el margen entre el primer y el quinto
  resultado (0,0244 en la consulta ambigua) sirve como señal de confianza.
- Sesión 06 (directo): RAG sobre el baseline de `evals/stress/REPORT.md`. Pendiente del directo de la 05: el
  patrón Actor-Critic-Boss (`evals/run.py --mode acb`), persistir las sesiones (Redis/Postgres) cuando haya más
  de un worker, streaming del turno con SSE y extractor LLM de `project_metadata` si la heurística se queda corta.
- Sesión 04: salida estructurada (JSON con las tareas y horas tipadas, no solo Markdown) para que el
  resultado sea tan consistente como la entrada; evaluación de versiones de prompt con las mismas entradas.
- Cacheo del prefijo estable del system prompt en el proveedor (ahora `v2` lo permite).
- Wrapper de abstracción de proveedores, cacheo de respuestas y capa de logging/trazabilidad (sesión 03).
- Mejorar los ejemplos de contexto (más variedad, casos parecidos a los reales).
- Evolucionar de CAG a RAG cuando los ejemplos no quepan en el contexto.
