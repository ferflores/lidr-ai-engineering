# Estimador CAG — Proyecto 1 (Sesiones 02, 03 y 04)

Servicio **FastAPI** que recibe la transcripción de una reunión con un cliente más unos **parámetros
tipados** (tarifa, tipo de proyecto, plazo, enfoque, nivel de detalle) y devuelve una **estimación de
software** (desglose de tareas, horas, coste, equipo y duración) generada por un LLM (**OpenAI** o
**Anthropic**). La interfaz es un **formulario en Streamlit** que muestra la estimación en streaming
(sesión 04; en la 03 era un chat). Los prompts son **plantillas versionadas** en `prompts/`, fuera del
código.

Arquitectura **CAG**: todo el contexto que necesita el modelo (estimaciones de ejemplo) viaja íntegro
dentro del prompt en cada llamada. No hay base de datos, ni retrieval, ni persistencia.

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
{ "estimation": "## Estimación: ...", "usage": {...}, "prompt": { "template": "estimacion", "version": "v2", ... } }
```

## Estructura

```
estimador-cag/
├── app/
│   ├── main.py                 # aplicación FastAPI, /health, Swagger
│   ├── config.py               # Settings (pydantic-settings) cargados desde .env, incluida PROMPT_VERSION
│   ├── schemas/estimation.py   # EstimationRequest: los parámetros tipados (contrato de API y formulario)
│   ├── routers/estimations.py  # POST /api/v1/estimate, GET /api/v1/prompts + schemas de respuesta
│   ├── services/prompts.py     # registro de plantillas: carga prompts/<nombre>/<versión>/ y las renderiza
│   ├── services/llm_service.py # renderiza la plantilla, inyecta los ejemplos y llama al LLM (sin texto de prompt)
│   ├── services/pricing.py     # precios por millón de tokens y cálculo del coste de cada llamada
│   └── context/examples.py     # estimaciones de ejemplo (el "conocimiento" del sistema)
├── prompts/                    # los prompts, fuera del código y versionados (ver prompts/README.md)
│   └── estimacion/
│       ├── v1/                 # template.toml + system.md.j2 + user.md.j2 (el prompt original, migrado)
│       └── v2/                 # parámetros tipados en el mensaje de usuario, system prompt estable (por defecto)
├── streamlit_app.py            # formulario con parámetros tipados, streaming, prompt enviado y métricas
├── .streamlit/                 # config.toml (headless) y secrets.toml.example (alternativa a .env)
├── tests/                      # pytest: API, plantillas, contexto, proveedores, streaming y el formulario (LLM simulado)
├── scripts/
│   ├── estimar.sh              # envía una transcripción (con o sin parámetros) al endpoint con curl
│   ├── comparar.py             # compara transcripciones: tokens, coste, ahorro y estimación
│   ├── generar-json.sh         # crea transcripciones/*.json a partir de los .txt
│   └── verificar.sh            # estructura + tests + arranque real (lo usa el CI)
├── transcripciones/            # transcripciones de ejemplo (.txt legible, .json listo para curl, *-parametros.json con parámetros)
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

Formulario web (ver "Formulario con parámetros tipados"):

```bash
uv run streamlit run streamlit_app.py    # http://localhost:8501
```

## Ejecutar con Docker

```bash
cp .env.example .env                 # rellena la API key (solo la primera vez)
docker compose up --build -d         # API en http://localhost:8000 y formulario en http://localhost:8501
docker compose logs -f api chat      # ver los logs
docker compose down                  # parar
```

Levanta dos servicios con la misma imagen: `api` (FastAPI, puerto 8000) y `chat` (el formulario de Streamlit, puerto 8501).

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
| `PROMPT_VERSION` | Versión de la plantilla de prompt por defecto (`prompts/estimacion/<versión>`) | `v2` |
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

## Formulario con parámetros tipados (Streamlit)

`streamlit_app.py`, en la raíz del proyecto, es la interfaz del estimador. En la sesión 03 era un chat;
en la 04 el chat deja de ser la interfaz principal y se sustituye por un **formulario** cuyos campos son
exactamente los de `EstimationRequest`, el mismo modelo que valida el endpoint.

```bash
uv run streamlit run streamlit_app.py      # abre http://localhost:8501
```

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

Tests sin llamadas reales: `tests/test_streaming.py` (el servicio de streaming) y
`tests/test_streamlit_app.py` (la app completa con `streamlit.testing.v1.AppTest`: abre un formulario y no
un chat, envía parámetros y comprueba lo que llega al LLM, la validación sin llamada, la carga de ejemplos
y el estado sin API key).

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
uv run pytest -q          # tests con el LLM sustituido por dobles (no gasta créditos)
scripts/verificar.sh      # estructura + tests + arranque real + llamada real si hay API key
```

`scripts/verificar.sh` comprueba, en orden:

1. Que existen todos los ficheros de la estructura del ejercicio (incluidas las plantillas `v1` y `v2`),
   que `.env` está en `.gitignore` y no versionado, que no hay API keys en el código, que el texto del
   prompt no está en el código y que `streamlit_app.py` no usa chat.
2. Que pasan los tests.
3. Que el servicio arranca y responden `GET /health` (200), `GET /docs` (200), que `POST /api/v1/estimate`
   está en el OpenAPI y valida el body tipado (422 con transcripción vacía, con un enumerado inválido y con
   una versión de plantilla inexistente), que `GET /api/v1/prompts` lista `v1` y `v2`, y que el formulario
   de Streamlit arranca.
4. Si hay API key configurada, que `POST /api/v1/estimate` con parámetros devuelve una estimación real.

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
- **Sin historial**: cada envío del formulario es una petición independiente y reproducible. La
  conversación con memoria de la sesión 03 se retiró junto con el chat.

## Próximos pasos (sesión en vivo)

- Sesión 04: salida estructurada (JSON con las tareas y horas tipadas, no solo Markdown) para que el
  resultado sea tan consistente como la entrada; evaluación de versiones de prompt con las mismas entradas.
- Cacheo del prefijo estable del system prompt en el proveedor (ahora `v2` lo permite).
- Wrapper de abstracción de proveedores, cacheo de respuestas y capa de logging/trazabilidad (sesión 03).
- Mejorar los ejemplos de contexto (más variedad, casos parecidos a los reales).
- Evolucionar de CAG a RAG cuando los ejemplos no quepan en el contexto.
