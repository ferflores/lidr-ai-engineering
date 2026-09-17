# Estimador CAG — Proyecto 1 (Sesiones 02 y 03)

Servicio **FastAPI** que recibe la transcripción de una reunión con un cliente y devuelve una
**estimación de software** (desglose de tareas, horas, coste, equipo y duración) generada por un LLM
(**OpenAI** o **Anthropic**), más una **interfaz de chat en Streamlit** que muestra la estimación en
streaming (sesión 03).

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
│   ├── services/pricing.py     # precios por millón de tokens y cálculo del coste de cada llamada
│   └── context/examples.py     # estimaciones de ejemplo (el "conocimiento" del sistema)
├── streamlit_app.py            # interfaz de chat (Streamlit) con streaming, historial y panel CAG
├── .streamlit/                 # config.toml (headless) y secrets.toml.example (alternativa a .env)
├── tests/                      # pytest: API, contexto, proveedores, streaming y la app de chat (LLM simulado)
├── scripts/
│   ├── estimar.sh              # envía una transcripción al endpoint con curl
│   ├── comparar.py             # compara transcripciones: tokens, coste, ahorro y estimación
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

Interfaz de chat (ver "Interfaz de chat con Streamlit"):

```bash
uv run streamlit run streamlit_app.py    # http://localhost:8501
```

## Ejecutar con Docker

```bash
cp .env.example .env                 # rellena la API key (solo la primera vez)
docker compose up --build -d         # API en http://localhost:8000 y chat en http://localhost:8501
docker compose logs -f api chat      # ver los logs
docker compose down                  # parar
```

Levanta dos servicios con la misma imagen: `api` (FastAPI, puerto 8000) y `chat` (Streamlit, puerto 8501).

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
| `LLM_TEMPERATURE` | Temperatura del modelo (0-2) | `0.2` |
| `LLM_MAX_TOKENS` | Máximo de tokens de salida | `4096` |
| `LLM_TIMEOUT_SECONDS` | Timeout de la llamada al proveedor | `60` |
| `LLM_INPUT_PRICE_PER_MTOK` | Precio en USD por millón de tokens de entrada (opcional, sobrescribe la tabla) | tabla de `pricing.py` |
| `LLM_OUTPUT_PRICE_PER_MTOK` | Precio en USD por millón de tokens de salida (opcional, sobrescribe la tabla) | tabla de `pricing.py` |

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
  "prompt": { "system_prompt_chars": 7458, "transcription_chars": 280 },
  "generated_at": "2026-09-16T17:10:10.585676Z"
}
```

- `usage.input_tokens` son los tokens del system prompt (contexto CAG) más la transcripción;
  `usage.output_tokens`, los de la estimación generada.
- `usage.cost` es el coste de la llamada según el precio de lista del modelo (ver "Tokens y coste").
- `prompt` da el tamaño en caracteres del contexto fijo y de la transcripción, para ver qué parte del
  prompt es contexto y qué parte es dato.

Códigos de respuesta: `200` estimación generada · `422` body inválido · `500` falta la API key · `502` el proveedor ha fallado.

## Interfaz de chat con Streamlit

`streamlit_app.py`, en la raíz del proyecto, es una interfaz conversacional sobre la misma lógica que el
endpoint: mismo system prompt, mismos ejemplos de contexto y misma llamada al LLM, pero en streaming.

```bash
uv run streamlit run streamlit_app.py      # abre http://localhost:8501
```

Qué hace:

- **Chat** con `st.chat_message` y `st.chat_input`. El historial se guarda en `st.session_state` y
  permanece en pantalla durante la sesión.
- **Streaming**: la estimación se va escribiendo fragmento a fragmento con `st.write_stream`, tanto con
  OpenAI (`stream=True` + `stream_options.include_usage`) como con Anthropic (`messages.stream`).
  La lógica está en `EstimationStream`, en `app/services/llm_service.py`.
- **Conversación con memoria**: por defecto cada mensaje nuevo incluye la conversación anterior, así se
  pueden pedir ajustes sobre la estimación ("reduce el alcance", "¿y sin el blog?"). El primer mensaje se
  trata como la transcripción y los siguientes como preguntas. Se puede desactivar en la barra lateral,
  y las métricas muestran cómo crecen los tokens de entrada con el historial.
- **Barra lateral (Nivel 3)**: proveedor (solo los que tienen API key), modelo y parámetros; el system
  prompt activo en solo lectura; el contexto estático inyectado (las estimaciones de ejemplo); ejemplos
  de transcripción para enviar con un clic; y las métricas de la última llamada: modelo, tokens de
  entrada, tokens de salida, tiempo y coste. Cada respuesta lleva además esos datos en su pie.
- **API key**: se lee de `.env` (igual que la API) o, si no está ahí, de `.streamlit/secrets.toml` con
  `st.secrets` (hay un `.streamlit/secrets.toml.example`). Nunca va en el código. Sin API key el chat
  aparece deshabilitado con un aviso.

Tests sin llamadas reales: `tests/test_streaming.py` (el servicio de streaming y el historial) y
`tests/test_streamlit_app.py` (la app completa con `streamlit.testing.v1.AppTest`: abre el chat, envía
una transcripción, comprueba el streaming, las métricas y que el segundo turno lleva el historial).

Checklist de la sesión 03:

- [x] `streamlit run streamlit_app.py` abre una interfaz de chat en el navegador
- [x] Puedes pegar una transcripción de reunión y recibes una estimación de software
- [x] La conversación persiste en pantalla (puedes hacer varias preguntas seguidas)
- [x] La respuesta se muestra en streaming, no de golpe
- [x] La API key se lee desde `.env` o `st.secrets`, no está en el código

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

En arquitectura CAG el system prompt con los ejemplos viaja **en todas las llamadas**: son unos 7.500
caracteres, alrededor de 1.900 tokens, y es la mayor parte de la entrada. La transcripción solo añade lo
que ocupe. Por eso, si el objetivo es abaratar, lo que pesa es el contexto fijo, no lo que escriba el cliente.

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
- **Campos extra en la respuesta**: `usage` (tokens y coste), `prompt` (tamaño del contexto fijo y de la
  transcripción) y `generated_at`. El coste se calcula en el servidor con una tabla de precios de lista
  para que cada llamada muestre lo que ha costado sin depender de la consola del proveedor.
- **Docker opcional**: el ejercicio no lo pide, pero el stack del curso corre en contenedores. El
  `Dockerfile` y el `docker-compose.yml` no cambian nada del código: la misma app corre con `uv run` o
  en contenedor.
- **Streaming síncrono para Streamlit**: Streamlit ejecuta el script de forma síncrona, así que
  `EstimationStream` usa los clientes síncronos de los SDK y expone un iterable que `st.write_stream`
  consume directamente. El endpoint REST sigue usando la versión asíncrona; ambos comparten el prompt,
  la traducción de errores y el cálculo de coste.
- **Historial en el chat**: el endpoint sigue siendo de un solo turno; la conversación con memoria vive
  solo en la interfaz, que decide qué turnos enviar.

## Próximos pasos (sesión en vivo)

- Sesión 03: wrapper de abstracción de proveedores, cacheo de respuestas y capa de logging/trazabilidad.
- Mejorar los ejemplos de contexto (más variedad, casos parecidos a los reales).
- Iterar el prompt y comparar la calidad de las estimaciones entre proveedores.
- Explorar parámetros del modelo (`LLM_TEMPERATURE`, `LLM_MAX_TOKENS`).
- Evolucionar de CAG a RAG cuando los ejemplos no quepan en el contexto.
